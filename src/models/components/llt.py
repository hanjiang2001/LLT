import re

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from torch import nn

from config.splits import load_class_names
from src.models.components.loss import LookupAlignmentLoss, SeparationLoss


def compute_confidence(proposals, activation, video_confidence, margin=0.25, gamma=0.2):
    activation = activation.detach().cpu().numpy()
    scores = []
    for start, end in proposals:
        if start > end:
            start, end = end, start
        length = end - start + 1
        inner_score = np.mean(activation[start : end + 1])
        outer_start = max(0, int(start - margin * length))
        outer_end = min(len(activation) - 1, int(end + margin * length))
        outer_indices = list(range(outer_start, start)) + list(range(end + 1, outer_end + 1))
        outer_score = np.mean(activation[outer_indices]) if outer_indices else 0
        scores.append(float(inner_score - outer_score + gamma * video_confidence))
    return scores


def update_lookup_table(lookup_table, label, feature, confidence, capacity):
    with torch.no_grad():
        items = lookup_table.setdefault(label, [])
        if len(items) < capacity:
            items.append((feature, confidence))
        elif confidence > items[-1][1]:
            items[-1] = (feature, confidence)
        items.sort(key=lambda item: item[1], reverse=True)


def update_text_prototype(prototypes, label, feature, confidence, momentum, threshold):
    if confidence > threshold:
        feature = feature.detach().cpu().numpy()
        prototypes[label] = momentum * prototypes[label] + (1 - momentum) * feature


class LookupResidual(nn.Module):
    def __init__(self, prototype):
        super().__init__()
        self.residual = nn.Parameter(torch.zeros_like(prototype, dtype=torch.float16))

    def forward(self, prototype):
        return F.normalize(prototype.clone() + self.residual, dim=1)


class LLTNet(nn.Module):
    def __init__(
        self,
        dataset,
        split,
        setting,
        checkpoint_path,
        stride,
        randper,
        kernel_size,
        n,
        normalize,
        remove_background,
        steps,
        lookup_capacity=4,
        text_momentum=0.6,
        update_threshold=0.7,
    ):
        super().__init__()
        self.dataset = dataset
        self.stride = stride
        self.randper = randper
        self.kernel_size = kernel_size
        self.n = n
        self.normalize = normalize
        self.remove_background = remove_background
        self.steps = steps
        self.lookup_capacity = lookup_capacity
        self.text_momentum = text_momentum
        self.update_threshold = update_threshold
        self.topk = 3
        self.m = 0.7
        self.class_names = load_class_names(dataset, setting, split)
        self.tokenize = open_clip.get_tokenizer("coca_ViT-L-14")
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name="coca_ViT-L-14", pretrained=checkpoint_path
        )
        self.model = self.model.float().eval()
        self.model.requires_grad_(False)
        with torch.no_grad():
            self.text_features = self.get_text_features()
        self.lookup_table = {}
        self.text_prototypes = {
            index: feature.cpu().numpy() for index, feature in enumerate(self.text_features)
        }
        self.separation_loss = SeparationLoss()
        self.alignment_loss = LookupAlignmentLoss()
        self.text_loss = SeparationLoss(smoothing=0.0)

    def get_text_features(self):
        prompts = [
            "a video of action " + re.sub("([a-z])([A-Z])", r"\1 \2", name)
            for name in self.class_names
        ]
        tokens = torch.stack([self.tokenize(prompt) for prompt in prompts]).squeeze(1)
        tokens = tokens.to(next(self.model.parameters()).device)
        features = self.model.encode_text(tokens)
        return features / features.norm(dim=-1, keepdim=True)

    def compute_score(self, x, y):
        x = x / x.norm(dim=-1, keepdim=True)
        return (self.model.logit_scale.exp() * x @ y.T).softmax(dim=-1)

    def select_segments(self, similarity):
        if self.dataset == "thumos":
            mask = similarity > similarity.mean()
        elif self.dataset == "anet":
            mask = similarity > self.m
        else:
            raise ValueError(f"Not implemented dataset: {self.dataset}")
        selected = torch.nonzero(mask).squeeze()
        segments = []
        if selected.numel() and selected.dim() > 0:
            interval_start = selected[0]
            for i in range(1, len(selected)):
                if selected[i] <= selected[i - 1] + self.stride:
                    continue
                else:
                    interval_end = selected[i - 1]
                    if interval_start != interval_end:
                        segments.append([interval_start.item(), interval_end.item()])
                    interval_start = selected[i]
            if interval_start != selected[-1]:
                segments.append([interval_start.item(), selected[-1].item()])
        return segments

    def infer_pseudo_labels(self, image_features):
        image_features_avg = image_features.mean(dim=0)
        self.background_embedding = image_features_avg.unsqueeze(0)
        self.text_features = self.text_features.to(image_features.device)
        scores_avg = self.compute_score(image_features_avg.unsqueeze(0), self.text_features)
        _, indexes = torch.topk(scores_avg, self.topk)
        return (indexes[0][0], scores_avg)

    def moving_average(self, data, window_size):
        padding_size = window_size
        padded_data = torch.cat(
            [
                torch.ones(padding_size).to(data.device) * data[0],
                data,
                torch.ones(padding_size).to(data.device) * data[-1],
            ]
        )
        kernel = (torch.ones(window_size) / window_size).to(data.device)
        smoothed_data = F.conv1d(padded_data.view(1, 1, -1), kernel.view(1, 1, -1))
        smoothed_data = smoothed_data.view(-1)[padding_size // 2 + 1 : -padding_size // 2]
        return smoothed_data

    def sample_visual_indices(self, signal):
        if 100 * self.n >= signal.shape[1]:
            pindices = torch.arange(signal.shape[1]).to(signal.device)
            nindices = torch.arange(signal.shape[1]).to(signal.device)
        else:
            pindices = torch.topk(signal, 100 * self.n % signal.shape[1])[1]
            nindices = torch.topk(-signal, 100 * self.n % signal.shape[1])[1]
        pindices = pindices.squeeze().sort()[0]
        nindices = nindices.squeeze().sort()[0]
        if pindices.shape[0] < self.n:
            pindices = pindices.repeat_interleave(self.n // pindices.shape[0] + 1)
            pindices = pindices[: self.n]
        if nindices.shape[0] < self.n:
            nindices = nindices.repeat_interleave(self.n // nindices.shape[0] + 1)
            nindices = nindices[: self.n]
        pindices = pindices[:: (len(pindices) - 1) // (self.n - 1)][: self.n]
        nindices = nindices[:: (len(nindices) - 1) // (self.n - 1)][: self.n]
        pindices = torch.clamp(
            pindices + torch.randint(-self.randper, self.randper, (self.n,)).to(signal.device),
            0,
            signal.shape[1] - 1,
        )
        nindices = torch.clamp(
            nindices + torch.randint(-self.randper, self.randper, (self.n,)).to(signal.device),
            0,
            signal.shape[1] - 1,
        )
        return (pindices, nindices)

    def sample_text_indices(self, signal):
        T = signal.shape[1]
        topk_num = max(1, int(T * 0.05))
        p_candidates = torch.topk(signal, k=topk_num, dim=1, largest=True)[1].flatten()
        n_candidates = torch.topk(-signal, k=topk_num, dim=1, largest=True)[1].flatten()
        p_candidates = torch.unique(p_candidates)
        n_candidates = torch.unique(n_candidates)

        def sample_or_repeat(indices, target_size):
            if indices.shape[0] < target_size:
                return indices.repeat(target_size // indices.shape[0] + 1)[:target_size]
            else:
                step = (indices.shape[0] - 1) // (target_size - 1)
                return indices[::step][:target_size]

        sample_count = 20
        pindices = sample_or_repeat(p_candidates.sort()[0], sample_count)
        nindices = sample_or_repeat(n_candidates.sort()[0], sample_count)
        pindices = torch.clamp(
            pindices
            + torch.randint(-self.randper, self.randper, (sample_count,), device=signal.device),
            0,
            T - 1,
        )
        nindices = torch.clamp(
            nindices
            + torch.randint(-self.randper, self.randper, (sample_count,), device=signal.device),
            0,
            T - 1,
        )
        return (pindices, nindices)

    def select_lookup_indices(self, signal):
        count = max(1, signal.shape[1] // 20)
        return torch.topk(signal, count, dim=-1)[1]

    def forward(self, batch):
        video_names, features_pre = batch
        video_name = video_names[0]
        with torch.no_grad():
            image_features = (features_pre @ self.model.visual.proj).squeeze(0)
            label, class_scores = self.infer_pseudo_labels(image_features)
            label = label.item()
            video_confidence = class_scores[0, label].item()
            print(f"{video_name}: {self.class_names[label]} ({video_confidence:.4f})")
            features = (
                image_features - self.background_embedding
                if self.remove_background
                else image_features
            )
            features_norm = features / features.norm(dim=-1, keepdim=True)
            text_prototype = (
                torch.from_numpy(self.text_prototypes[label])
                .to(image_features.device)
                .reshape(1, -1)
            )
            text_prototype = text_prototype / text_prototype.norm(dim=-1, keepdim=True)
            activation = self.model.logit_scale.exp() * text_prototype @ features_norm.T
            if self.dataset == "thumos":
                activation = self.moving_average(activation.squeeze(), self.kernel_size)
            if self.normalize:
                activation = (activation - activation.min()) / (activation.max() - activation.min())
            positive_indices = self.select_lookup_indices(activation.squeeze().unsqueeze(0))
            lookup_candidate = features_norm[positive_indices].squeeze(0).mean(dim=0)
            lookup_candidate = lookup_candidate / lookup_candidate.norm(dim=-1, keepdim=True)
            update_lookup_table(
                self.lookup_table,
                label,
                lookup_candidate.unsqueeze(0),
                video_confidence,
                self.lookup_capacity,
            )
            lookup_item = (
                torch.stack([item[0] for item in self.lookup_table[label]])
                .squeeze(1)
                .mean(dim=0)
                .unsqueeze(0)
            )
            lookup_item = lookup_item / lookup_item.norm(dim=-1, keepdim=True)

        lookup_residual = LookupResidual(lookup_item)
        lookup_optimizer = torch.optim.AdamW(
            [{"params": lookup_residual.parameters(), "lr": 0.001, "eps": 0.001}]
        )
        for _ in range(self.steps):
            adapted_item = lookup_residual(lookup_item)
            activation = self.model.logit_scale.exp() * adapted_item @ features_norm.T
            if self.dataset == "thumos":
                activation = self.moving_average(activation.squeeze(), self.kernel_size).unsqueeze(
                    0
                )
            positive_indices, negative_indices = self.sample_visual_indices(activation)
            positive_features = image_features[positive_indices]
            negative_features = image_features[negative_indices]
            positive_features = positive_features / positive_features.norm(dim=-1, keepdim=True)
            negative_features = negative_features / negative_features.norm(dim=-1, keepdim=True)
            positive_scores = self.model.logit_scale.exp() * adapted_item @ positive_features.T
            negative_scores = self.model.logit_scale.exp() * adapted_item @ negative_features.T
            scores = torch.cat([positive_scores.squeeze(), negative_scores.squeeze()], dim=0)
            targets = torch.cat(
                [torch.ones(positive_scores.shape[1]), torch.zeros(negative_scores.shape[1])]
            ).to(scores.device)
            loss = self.separation_loss(scores, targets) + 0.5 * self.alignment_loss(
                adapted_item, positive_features
            )
            lookup_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            lookup_optimizer.step()

        adapted_item = adapted_item.detach()
        text_residual = LookupResidual(text_prototype)
        text_optimizer = torch.optim.AdamW(
            [{"params": text_residual.parameters(), "lr": 0.003, "eps": 0.001}]
        )
        for _ in range(self.steps):
            adapted_text = text_residual(text_prototype)
            activation = self.model.logit_scale.exp() * adapted_item @ features_norm.T
            if self.dataset == "thumos":
                activation = self.moving_average(activation.squeeze(), self.kernel_size).unsqueeze(
                    0
                )
            positive_indices, negative_indices = self.sample_text_indices(activation)
            positive_features = image_features[positive_indices]
            negative_features = image_features[negative_indices]
            positive_features = positive_features / positive_features.norm(dim=-1, keepdim=True)
            negative_features = negative_features / negative_features.norm(dim=-1, keepdim=True)
            positive_scores = self.model.logit_scale.exp() * adapted_text @ positive_features.T
            negative_scores = self.model.logit_scale.exp() * adapted_text @ negative_features.T
            scores = torch.cat([positive_scores.squeeze(), negative_scores.squeeze()], dim=0)
            targets = torch.cat(
                [torch.ones(positive_scores.shape[1]), torch.zeros(negative_scores.shape[1])]
            ).to(scores.device)
            loss = self.text_loss(scores, targets)
            text_optimizer.zero_grad()
            loss.backward()
            text_optimizer.step()

        with torch.no_grad():
            adapted_text = text_residual(text_prototype)
            update_text_prototype(
                self.text_prototypes,
                label,
                adapted_text.reshape(-1),
                video_confidence,
                self.text_momentum,
                self.update_threshold,
            )
            activation = self.model.logit_scale.exp() * adapted_item @ features_norm.T
            if self.dataset == "thumos":
                activation = self.moving_average(activation.squeeze(), self.kernel_size)
            if self.normalize:
                activation = (activation - activation.min()) / (activation.max() - activation.min())
            activation = activation.squeeze()
            segments = self.select_segments(activation)
            confidences = compute_confidence(segments, activation, video_confidence)
            predictions = [
                {"label": label, "score": score, "segment": segment}
                for segment, score in zip(segments, confidences)
            ]
        return video_name, predictions

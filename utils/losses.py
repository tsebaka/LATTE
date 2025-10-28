import torch
from congpt.utils.ptls_extensions import PaddedBatch
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07, lambda_orth: float = 0.0):
        """
        Args:
            temperature: scaling factor for contrastive logits
            lambda_orth: weight for optional orthogonality regularization
        """
        super().__init__()
        self.temperature = temperature
        self.lambda_orth = lambda_orth

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Symmetric InfoNCE loss with optional orthogonality penalty between z1 and z2.

        Args:
            z1: Tensor of shape (B, D)
            z2: Tensor of shape (B, D)
        Returns:
            Scalar loss
        """
        B, D = z1.shape
        assert z2.shape == (B, D), "z1 and z2 must have same shape"

        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # ----- InfoNCE -----
        logits_12 = torch.matmul(z1, z2.T) / self.temperature
        logits_21 = logits_12.T
        labels = torch.arange(B, device=z1.device)

        loss_12 = F.cross_entropy(logits_12, labels)
        loss_21 = F.cross_entropy(logits_21, labels)
        loss = (loss_12 + loss_21) / 2

        # ----- Optional orthogonality penalty -----
        if self.lambda_orth > 0:
            M = z1.T @ z2
            orth_penalty = (M.pow(2).sum())
            loss = loss + self.lambda_orth * orth_penalty

        return loss


class PairwiseMSELoss(nn.Module):
    def __init__(self, normalize: bool = False, reduction: str = "mean"):
        """
        Args:
            normalize: если True — L2-нормализует эмбеддинги перед подсчётом MSE
            reduction: "mean" | "sum" | "none"
        """
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.normalize = normalize
        self.reduction = reduction

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: (B, D)
            z2: (B, D)
        Returns:
            Скалярный лосс (если reduction != "none") или тензор (B,) при reduction="none"
        """
        B, D = z1.shape
        assert z2.shape == (B, D), "z1 и z2 должны иметь одинаковую форму"

        if self.normalize:
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)

        # Поэлементный MSE, затем усредняем по признакам → получаем лосс на сэмпл
        per_sample = (z1 - z2).pow(2).mean(dim=1)  # (B,)

        if self.reduction == "mean":
            return per_sample.mean()
        elif self.reduction == "sum":
            return per_sample.sum()
        else:  # "none"
            return per_sample


def recursive_to(collection, *args, **kwargs):
    if isinstance(collection, (torch.Tensor, torch.nn.Module, PaddedBatch)):
        return collection.to(*args, **kwargs)
    if isinstance(collection, dict):
        return {k: recursive_to(v, *args, **kwargs) for k, v in collection.items()}
    if isinstance(collection, (tuple, list)):
        return [recursive_to(v, *args, **kwargs) for v in collection]
    raise ValueError(f"Unknown collection type {type(collection)}.")


def truncate_batch(tensor, truncation):
    if truncation == 0:
        return tensor
    assert truncation > 0
    if isinstance(tensor, torch.Tensor):
        offset = truncation // 2
        right_offset = truncation - offset
        return tensor[:, offset:tensor.shape[1] - right_offset]
    elif isinstance(tensor, dict):
        return {k: truncate_batch(v, truncation) for k, v in tensor.items()}
    assert isinstance(tensor, PaddedBatch)
    return PaddedBatch(truncate_batch(tensor.payload, truncation), (tensor.seq_lens - truncation).clip(min=0))


class ScaleGradient(torch.autograd.Function):
    """Scale gradient."""

    @staticmethod
    def forward(ctx, src, weight):
        ctx._weight = weight
        return src

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx._weight, None
    
class MSEMLELoss(torch.nn.Module):
    """MSE for continuous features, compatible with MLELoss."""

    __constants__ = ["input_dim"]
    
    def __init__(self, weight=1.0):
        super().__init__()
        self.input_dim = 1  
        self._weight = weight

    def get_means(self, input):
        return input

    def get_modes(self, input):
        return input

    def get_sample(self, input, temperature=1):
        
        if temperature != 1:
            scale = temperature
        else:
            scale = 1.0
        noise = torch.randn_like(input) * scale
        return input + noise

    def logpdf(self, input, target):
        diff = (input - target).squeeze(-1)
        raw = - diff.pow(2)
        return ScaleGradient.apply(raw, self._weight)

    def forward(self, input, target, masks):
        # input: (B, T, 1), target: (B, T, 1), masks: (B, T)
        pred = input.squeeze(2)
        tgt  = target.squeeze(2)
        errors = (pred - tgt).pow(2)  # (B, T)
        loss = (errors * masks).sum() / masks.sum()
        return ScaleGradient.apply(loss, self._weight)


class CatMLELoss(torch.nn.Module):
    """NLL for categorical features (cross-entropy)."""

    __constants__ = ["input_dim", "target_dim"]
    input_dim: int
    target_dim: int

    def __init__(self, num_classes, weight=1):
        super().__init__()
        self.input_dim = num_classes
        self.target_dim = 1
        self._weight = weight

    def get_means(self, input):
        # Means for categorical distributions are not defined. Return modes instead.
        return torch.argmax(input, dim=-1, keepdim=True)  # (*, 1).

    def get_modes(self, input):
        return torch.argmax(input, dim=-1, keepdim=True)  # (*, 1).

    def get_sample(self, input, temperature=1):
        logits = input / temperature if temperature != 1 else input
        probs = torch.nn.functional.softmax(logits, dim=-1)
        nc = probs.shape[-1]
        out_shape = list(probs.shape[:-1]) + [1]
        return torch.multinomial(probs.reshape(-1, nc), 1, replacement=True).reshape(out_shape)  # (..., 1).

    def logpdf(self, input, target):
        # input: (*, C).
        # target: (*, 1).
        lognorms = torch.logsumexp(input, dim=-1)  # (*).
        gt_logits = input.take_along_dim(target.long(), -1).squeeze(-1)  # (*).
        return ScaleGradient.apply(gt_logits - lognorms, self._weight)  # (*).

    def forward(self, input, target, masks):
        # input: (B, T, C).
        # target: (B, T, 1).
        # masks: (B, T).
        target = target.squeeze(2).long()
        assert target.ndim == 2, target.shape
        xents = torch.nn.functional.cross_entropy(input.permute(0, 2, 1), target, reduction="none")  # (B, T).
        xent = (xents * masks).sum() / masks.sum()
        return ScaleGradient.apply(xent, self._weight)


class MLELoss(torch.nn.Module):
    """Works like `ptls.loss.MultiLoss`

    Parameters
    ----------
    losses:
      mapping from feature name to MLE loss.

    """
    def __init__(self, losses):
        super().__init__()
        self.losses = torch.nn.ModuleDict(losses)
        self._stride = 1

    @property
    def loss_names(self):
        return list(sorted(list(self.losses)))

    @property
    def stride(self):
        return self._stride

    @stride.setter
    def stride(self, value):
        if value < 1 - 1e-6:
            raise RuntimeError(f"Dilated autoencoder with stride {value}.")
        if abs(value - round(value)) > 1e-6:
            raise RuntimeError("Non-integer autoencoder stride.")
        self._stride = int(round(value))

    @property
    def input_dim(self):
        return {name: self.losses[name].input_dim for name in self.loss_names}

    def compute_head_losses(self, head_outputs, targets):
        losses = {}
        for name, output in head_outputs.payload.items():
            target = targets.payload[name]
            if target.ndim == 2:
                target = target.unsqueeze(2)
            losses[name] = self.losses[name](output, target, targets.seq_len_mask)
        return {
            "loss": losses,
            "log": {}
        }

    def compute_all_losses_impl(self, model_outputs, targets):
        # Align predictions with targets and compute head outputs.
        result = defaultdict(dict)
        for head_name, head_outputs in model_outputs.items():
            if not isinstance(head_outputs.payload, dict):
                raise ValueError("Need dictionary with outputs for each target.")
            unused_outputs = set(head_outputs.payload) - set(self.losses)
            if unused_outputs:
                raise RuntimeError(f"Some targets are unused: {unused_outputs}.")
            aligned_outputs, aligned_targets = self._align_predictions_targets(head_outputs, targets)
            head_outputs = self.compute_head_losses(aligned_outputs, aligned_targets)
            for output_name, outputs in head_outputs.items():
                if isinstance(outputs, dict):
                    for name, value in outputs.items():
                        name = f"{head_name}:{name}" if head_name else name
                        result[output_name][name] = value
                else:
                    name = f"{head_name}:{output_name}" if head_name else output_name
                    result[name] = outputs
        return dict(result)

    def compute_all_losses(self, model_outputs, targets):
        """Compute losses.

        Args:
            model_outputs: Nested dict head->target->Tensor.
            target: Dict target->Tensor.

        Returns:
            Dict with dict of losses and dict of log values.
        """
        if not model_outputs:
            raise ValueError("Empty model outputs")

        # Compute losses in full precision.
        model_outputs = recursive_to(model_outputs, torch.float32)
        targets = recursive_to(targets, torch.float32)

        with torch.autocast(targets.device.type,
                            dtype=torch.float32 if targets.device.type != "cpu" else torch.bfloat16,
                            enabled=targets.device.type != "cpu"):
            return self.compute_all_losses_impl(model_outputs, targets)

    def forward(self, model_outputs, targets):
        """Compute losses.

        Args:
            model_outputs: Nested dict head->target->Tensor.
            target: Dict target->Tensor.

        Returns:
            Average loss.
        """
        result = self.compute_all_losses(model_outputs, targets)
        result["loss"] = sum(result["loss"].values()) / len(result["loss"])
        return result

    def logpdf(self, parameters, targets, loss_weights=None):
        """Compute targets density according to predicted distributions.

        Args:
          parameters: Predicted distributions parameters with shape (*, P).
          targets: Target values with shape (*, D).
          loss_weights: A dict of loss weights. If provided, use only losses from this dict.

        Returns:
          PDFs with shape (*).
        """
        if loss_weights is None:
            loss_weights = {name: 1 for name in self.loss_names}
        loss_logpdfs = [self.losses[name].logpdf(parameters[name], targets[name]) * weight
                        for name, weight in loss_weights.items()]
        logpdfs = torch.stack(loss_logpdfs).sum(0)  # (*).
        return logpdfs

    def get_means(self, model_outputs):
        """Get distribution means (modes when undefined).

        Args:
            model_outputs: Dict target->Tensor.

        Returns:
            Dictionary with means.
        """
        means = {}
        for target_name, loss in self.losses.items():
            means[target_name] = loss.get_means(model_outputs[target_name])
        return means

    def get_modes(self, model_outputs):
        """Get distribution modes.

        Args:
            model_outputs: Dict target->Tensor.

        Returns:
            Dictionary with modes.
        """
        modes = {}
        for target_name, loss in self.losses.items():
            modes[target_name] = loss.get_modes(model_outputs[target_name])
        return modes

    def get_sample(self, model_outputs, temperature=1):
        """Get distribution sample.

        Args:
            model_outputs: Dict target->Tensor.

        Returns:
            Dictionary with sample of the same size, as inputs.
        """
        samples = {}
        for target_name, loss in self.losses.items():
            samples[target_name] = loss.get_sample(model_outputs[target_name], temperature)
        return samples

    def _align_predictions_targets(self, model_outputs, targets):
        """Align model_outputs and targets if they have different lengths."""
        # Adjust targets stride.
        targets, lengths = targets.payload, targets.seq_lens
        targets = {k: v[:, ::self._stride] for k, v in targets.items()}
        lengths = (lengths - 1) // self._stride + 1
        targets = PaddedBatch(targets, lengths)

        delta = model_outputs.seq_feature_shape[1] - targets.seq_feature_shape[1]
        if delta > 0:
            # More outputs than targets.
            model_outputs = truncate_batch(model_outputs, delta)
        elif delta < 0:
            # More targets than outputs.
            targets = truncate_batch(targets, abs(delta))

        return model_outputs, targets
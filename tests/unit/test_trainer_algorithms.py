"""Tensor-level proof that each FL algorithm actually differs from FedAvg.

Follows the D-020 precedent set for FedProx: it is not enough that a code path
exists and runs. Every algorithm must take at least two local optimiser steps
and produce weights that differ from a plain FedAvg run under an identical
seed. Without this, a silently-degraded implementation still yields a
plausible-looking results table.
"""

from __future__ import annotations

import importlib.util
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _tiny_model(seed: int = 0):
    """A miniature torchvision-shaped backbone: ``features`` + ``classifier``."""

    import torch

    torch.manual_seed(seed)

    class _TinyNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(4),
                torch.nn.ReLU(),
            )
            self.classifier = torch.nn.Linear(4, 3)

        def forward(self, x):
            pooled = torch.nn.functional.adaptive_avg_pool2d(self.features(x), (1, 1))
            return self.classifier(torch.flatten(pooled, 1))

    return _TinyNet()


def _tiny_loader(batches: int = 3, batch_size: int = 4, seed: int = 1):
    """A deterministic in-memory loader with a real ``dataset`` length."""

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    generator = torch.Generator().manual_seed(seed)
    total = batches * batch_size
    images = torch.randn(total, 3, 8, 8, generator=generator)
    labels = torch.randint(0, 3, (total,), generator=generator)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def _zeros_like_trainable(model):
    import torch

    return {
        name: torch.zeros_like(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _final_weights(result_model) -> dict:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in result_model.named_parameters()
    }


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class ScaffoldTrainerTests(unittest.TestCase):
    def test_scaffold_changes_weights_relative_to_fedavg(self) -> None:
        """A non-zero (c_i - c) must steer the update away from FedAvg."""

        import torch

        from cropfed.ml.trainer import set_reproducible_seed, train_local

        set_reproducible_seed(7)
        baseline_model = _tiny_model()
        baseline = train_local(
            baseline_model,
            _tiny_loader(),
            epochs=2,
            learning_rate=0.05,
            device=torch.device("cpu"),
        )

        set_reproducible_seed(7)
        scaffold_model = _tiny_model()
        server_c = _zeros_like_trainable(scaffold_model)
        client_c = {
            name: torch.full_like(value, 0.15)
            for name, value in _zeros_like_trainable(scaffold_model).items()
        }
        scaffold = train_local(
            scaffold_model,
            _tiny_loader(),
            epochs=2,
            learning_rate=0.05,
            device=torch.device("cpu"),
            scaffold_control_variate=client_c,
            scaffold_server_c=server_c,
        )

        self.assertIsNone(baseline.scaffold_c_i)
        self.assertIsNotNone(scaffold.scaffold_c_i)

        baseline_weights = _final_weights(baseline_model)
        scaffold_weights = _final_weights(scaffold_model)
        differing = [
            name
            for name, value in scaffold_weights.items()
            if not torch.allclose(value, baseline_weights[name], atol=1e-7)
        ]
        self.assertTrue(
            differing,
            "SCAFFOLD produced weights identical to FedAvg — the control "
            "variate correction was not applied",
        )

    def test_scaffold_control_variate_uses_local_step_count(self) -> None:
        """c_i⁺ must be scaled by optimiser steps, not by the example count."""

        import torch

        from cropfed.ml.trainer import _compute_scaffold_c_i

        model = _tiny_model()
        initial = {
            name: parameter.detach().clone() + 0.1
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        zeros = _zeros_like_trainable(model)
        few = _compute_scaffold_c_i(
            model, initial, zeros, zeros, local_steps=2, learning_rate=0.5
        )
        many = _compute_scaffold_c_i(
            model, initial, zeros, zeros, local_steps=8, learning_rate=0.5
        )
        name = next(iter(few))
        # step term = (w0 - w1) / (K·η) → 4× more steps ⇒ 4× smaller term.
        torch.testing.assert_close(few[name], many[name] * 4.0)

    def test_scaffold_rejects_mismatched_control_variate(self) -> None:
        import torch

        from cropfed.ml.trainer import train_local

        model = _tiny_model()
        server_c = _zeros_like_trainable(model)
        client_c = _zeros_like_trainable(model)
        client_c.pop(next(iter(client_c)))

        with self.assertRaisesRegex(ValueError, "missing c_i"):
            train_local(
                model,
                _tiny_loader(),
                epochs=1,
                learning_rate=0.05,
                device=torch.device("cpu"),
                scaffold_control_variate=client_c,
                scaffold_server_c=server_c,
            )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class MoonTrainerTests(unittest.TestCase):
    def test_moon_changes_weights_and_reports_contrastive_loss(self) -> None:
        import torch

        from cropfed.ml.trainer import set_reproducible_seed, train_local

        set_reproducible_seed(11)
        baseline_model = _tiny_model()
        baseline = train_local(
            baseline_model,
            _tiny_loader(),
            epochs=2,
            learning_rate=0.05,
            device=torch.device("cpu"),
        )
        self.assertIsNone(baseline.moon_contrastive_loss)

        set_reproducible_seed(11)
        moon_model = _tiny_model()
        # Distinct references, otherwise both similarities coincide and the
        # contrastive gradient vanishes for reasons unrelated to the code.
        previous_model = _tiny_model(seed=3)
        global_model = _tiny_model(seed=0)
        moon = train_local(
            moon_model,
            _tiny_loader(),
            epochs=2,
            learning_rate=0.05,
            device=torch.device("cpu"),
            moon_previous_model=previous_model,
            moon_global_model=global_model,
            moon_temperature=0.5,
            moon_mu=1.0,
        )

        self.assertIsNotNone(moon.moon_contrastive_loss)
        self.assertGreater(moon.moon_contrastive_loss, 0.0)

        baseline_weights = _final_weights(baseline_model)
        moon_weights = _final_weights(moon_model)
        differing = [
            name
            for name, value in moon_weights.items()
            if not torch.allclose(value, baseline_weights[name], atol=1e-7)
        ]
        self.assertTrue(
            differing,
            "MOON produced weights identical to FedAvg — the contrastive term "
            "was not applied",
        )

    def test_moon_representation_keeps_gradient(self) -> None:
        """The current model's representation must stay in the autograd graph.

        A detached representation makes the contrastive loss a constant, so
        training silently reduces to FedAvg while still reporting a loss value.
        """

        import torch

        from cropfed.ml.trainer import _extract_representation

        images = torch.randn(2, 3, 8, 8)

        features_model = _tiny_model()
        representation = _extract_representation(features_model, images)
        self.assertTrue(
            representation.requires_grad,
            "representation from a 'features' backbone is detached",
        )

        class _ResNetLike(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.body = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 4, 3, padding=1),
                    torch.nn.AdaptiveAvgPool2d((1, 1)),
                    torch.nn.Flatten(),
                )
                self.fc = torch.nn.Linear(4, 3)

            def forward(self, x):
                return self.fc(self.body(x))

        fc_representation = _extract_representation(_ResNetLike(), images)
        self.assertTrue(
            fc_representation.requires_grad,
            "representation captured from the 'fc' hook is detached, so MOON "
            "would contribute no gradient on ResNet backbones",
        )

    def test_moon_rejects_backbone_without_representation(self) -> None:
        import torch

        from cropfed.ml.trainer import _extract_representation

        class _Opaque(torch.nn.Module):
            def forward(self, x):  # pragma: no cover - never reached
                return x

        with self.assertRaisesRegex(TypeError, "features.*fc|fc.*features"):
            _extract_representation(_Opaque(), torch.randn(1, 3, 8, 8))


if __name__ == "__main__":
    unittest.main()

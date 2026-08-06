import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from cropfed.constants import TOMATO_CLASS_GROUPS, TOMATO_CLASSES

TORCH_RUNTIME_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "torchvision")
)


@unittest.skipUnless(
    TORCH_RUNTIME_AVAILABLE,
    "full Torch/Torchvision runtime is not installed",
)
class TorchRuntimeTests(unittest.TestCase):
    def test_fedprox_uses_squared_l2_distance(self) -> None:
        import torch

        from cropfed.ml.trainer import squared_l2_distance_to_reference

        model = torch.nn.Linear(2, 1, bias=True)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[3.0, 4.0]]))
            model.bias.copy_(torch.tensor([2.0]))
        reference = {
            "weight": torch.zeros_like(model.weight),
            "bias": torch.ones_like(model.bias),
        }

        distance = squared_l2_distance_to_reference(model, reference)

        self.assertAlmostEqual(float(distance.detach()), 26.0)
        self.assertAlmostEqual(
            float(((0.2 / 2.0) * distance).detach()),
            2.6,
            places=6,
        )

    def test_tiny_torch_train_and_evaluate(self) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from cropfed.ml.trainer import evaluate_model, train_local

        torch.manual_seed(2026)
        features = torch.randn(20, 4)
        labels = torch.arange(20) % 10
        loader = DataLoader(
            TensorDataset(features, labels),
            batch_size=5,
            shuffle=False,
        )
        model = torch.nn.Linear(4, 10)

        training = train_local(
            model,
            loader,
            epochs=2,
            learning_rate=0.01,
            device=torch.device("cpu"),
            proximal_mu=0.1,
        )
        evaluation = evaluate_model(
            model,
            loader,
            device=torch.device("cpu"),
            class_names=TOMATO_CLASSES,
            class_groups=TOMATO_CLASS_GROUPS,
        )

        self.assertEqual(training.num_examples, 20)
        self.assertGreaterEqual(training.loss, 0.0)
        self.assertEqual(evaluation.num_examples, 20)
        self.assertIn("macro_f1", evaluation.metrics)
        self.assertEqual(len(evaluation.metrics["confusion_matrix"]), 10)

    def test_validation_training_restores_selected_epoch(self) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from cropfed.ml.trainer import evaluate_model, train_with_validation

        torch.manual_seed(7)
        features = torch.randn(30, 4)
        labels = torch.arange(30) % 10
        train_loader = DataLoader(
            TensorDataset(features[:20], labels[:20]),
            batch_size=5,
            shuffle=False,
        )
        validation_loader = DataLoader(
            TensorDataset(features[20:], labels[20:]),
            batch_size=5,
            shuffle=False,
        )
        model = torch.nn.Linear(4, 10)

        result = train_with_validation(
            model,
            train_loader,
            validation_loader,
            epochs=3,
            learning_rate=0.01,
            device=torch.device("cpu"),
            class_names=TOMATO_CLASSES,
            class_groups=TOMATO_CLASS_GROUPS,
        )
        restored = evaluate_model(
            model,
            validation_loader,
            device=torch.device("cpu"),
            class_names=TOMATO_CLASSES,
            class_groups=TOMATO_CLASS_GROUPS,
        )

        self.assertEqual(len(result.history), 3)
        self.assertIn(result.best_epoch, {1, 2, 3})
        self.assertAlmostEqual(
            restored.metrics["macro_f1"],
            result.best_validation.metrics["macro_f1"],
        )
        self.assertAlmostEqual(restored.loss, result.best_validation.loss)

    def test_versioned_checkpoint_and_local_inference(self) -> None:
        from cropfed.ml.checkpoint import (
            CHECKPOINT_FORMAT_VERSION,
            load_model_checkpoint,
            save_model_checkpoint,
        )
        from cropfed.ml.inference import predict_image
        from cropfed.ml.model import build_model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            image_path = root / "leaf.png"
            Image.new("RGB", (48, 48), (42, 117, 61)).save(image_path)
            model = build_model("mobilenet_v2", num_classes=10, pretrained=False)
            checkpoint_info = save_model_checkpoint(
                checkpoint,
                model,
                model_name="mobilenet_v2",
                metadata={"seed": 2026, "experiment_type": "tiny-runtime-test"},
                class_order=TOMATO_CLASSES,
            )

            loaded = load_model_checkpoint(checkpoint)
            prediction = predict_image(
                checkpoint_path=checkpoint,
                image_path=image_path,
                top_k=3,
            )

        self.assertEqual(loaded.format_version, CHECKPOINT_FORMAT_VERSION)
        self.assertEqual(loaded.metadata["seed"], 2026)
        self.assertEqual(len(checkpoint_info["sha256"]), 64)
        self.assertGreater(checkpoint_info["bytes"], 0)
        self.assertEqual(prediction["crop"], "Tomato")
        self.assertIn(prediction["predicted_group"], {"healthy", "disease", "pest"})
        self.assertEqual(prediction["model"], "mobilenet_v2")
        self.assertEqual(
            prediction["checkpoint_format_version"],
            CHECKPOINT_FORMAT_VERSION,
        )
        self.assertEqual(len(prediction["predictions"]), 3)
        self.assertFalse(prediction["image_uploaded"])

    def test_checkpoint_state_comparison_ignores_envelope_metadata(self) -> None:
        import torch

        from cropfed.flower.smoke import compare_checkpoint_states
        from cropfed.ml.checkpoint import save_model_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "first.pt"
            second_path = root / "second.pt"
            first = torch.nn.Linear(2, 10)
            second = torch.nn.Linear(2, 10)
            second.load_state_dict(first.state_dict())
            with torch.no_grad():
                second.weight[0, 0] += 0.25
            save_model_checkpoint(
                first_path, first, model_name="tiny", class_order=TOMATO_CLASSES
            )
            save_model_checkpoint(
                second_path, second, model_name="tiny", class_order=TOMATO_CLASSES
            )

            comparison = compare_checkpoint_states(first_path, second_path)

        self.assertEqual(comparison["tensor_count"], 2)
        self.assertEqual(comparison["different_tensors"], 1)
        self.assertEqual(comparison["different_values"], 1)
        self.assertAlmostEqual(comparison["max_abs_difference"], 0.25)
        self.assertAlmostEqual(comparison["l2_distance"], 0.25)


if __name__ == "__main__":
    unittest.main()

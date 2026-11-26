"""Unit and integration tests for Set 10: CLIP Zero-Shot Classification."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.clip_core import (
    CLIPModelLoadError,
    DEFAULT_CLIP_MODEL,
    DEFAULT_CLIP_TEMPLATE,
    classify_image_zero_shot,
    clear_clip_cache,
    get_cached_clip_model,
    load_clip_model,
    load_rgb_image,
    render_clip_zero_shot_plot,
    validate_class_entries,
)
from forensics_app.tools.clip_tools import CLIPZeroShotTool


class _FakeCLIPModel:
    """Minimal PyTorch-compatible fake CLIP model for offline unit testing."""

    def __init__(self) -> None:
        self.eval_called = False
        self.logit_scale = torch.tensor(np.log(10.0), dtype=torch.float32)

    def to(self, _device: str) -> "_FakeCLIPModel":
        return self

    def eval(self) -> "_FakeCLIPModel":
        self.eval_called = True
        return self

    def get_image_features(self, **_kwargs) -> torch.Tensor:
        return torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    def get_text_features(self, input_ids: torch.Tensor, **_kwargs) -> torch.Tensor:
        # Return distinct embeddings per prompt index so ranking is deterministic
        n = input_ids.shape[0]
        feats = []
        for i in range(n):
            vec = [0.0, 0.0, 0.0, 0.0]
            # Prompt 1 ("dog" or custom) gets highest alignment with [1, 0, 0, 0]
            if i == 1:
                vec[0] = 2.0
                vec[1] = 0.1
            elif i == 0:
                vec[0] = 0.5
                vec[1] = 1.0
            else:
                vec[2] = 1.0
            feats.append(vec)
        return torch.tensor(feats, dtype=torch.float32)


class _FakeCLIPProcessor:
    def __call__(self, images=None, text=None, return_tensors="pt", padding=True):
        if images is not None:
            return {"pixel_values": torch.zeros((len(images), 3, 224, 224), dtype=torch.float32)}
        return {"input_ids": torch.arange(len(text), dtype=torch.long).unsqueeze(1)}


class Set10RegistryTests(unittest.TestCase):
    def test_clip_tool_registered(self) -> None:
        registry = build_tool_registry()
        cat_map = dict(registry.categories())
        self.assertIn("Set 10: CLIP Zero-Shot Classification", cat_map)
        tools = cat_map["Set 10: CLIP Zero-Shot Classification"]
        self.assertEqual([t.tool_id for t in tools], ["clip_zero_shot"])
        self.assertIsInstance(tools[0], CLIPZeroShotTool)


class Set10ValidationTests(unittest.TestCase):
    def test_validate_class_entries_formats_default_and_custom_prompts(self) -> None:
        pairs = validate_class_entries(
            [("cat", ""), ("dog", "a closeup of a happy puppy"), "coffee"],
            template=DEFAULT_CLIP_TEMPLATE,
        )
        self.assertEqual(
            pairs,
            [
                ("cat", "a photo of a cat."),
                ("dog", "a closeup of a happy puppy"),
                ("coffee", "a photo of a coffee."),
            ],
        )

    def test_validate_rejects_empty_blank_and_duplicate_inputs(self) -> None:
        with self.assertRaises(ValueError):
            validate_class_entries([])
        with self.assertRaises(ValueError):
            validate_class_entries([("   ", "")])
        with self.assertRaises(ValueError):
            validate_class_entries([("cat", "   ")])
        with self.assertRaises(ValueError):
            validate_class_entries([("cat", ""), ("CAT", "")])
        with self.assertRaises(ValueError):
            validate_class_entries(["cat"], template="   ")

    def test_load_rgb_image_validates_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            valid_path = tmp_path / "sample.png"
            Image.new("RGBA", (16, 12), (255, 0, 0, 128)).save(valid_path)
            rgb, name = load_rgb_image(valid_path)
            self.assertEqual(rgb.mode, "RGB")
            self.assertEqual(name, "sample.png")

            bad_ext = tmp_path / "notes.txt"
            bad_ext.write_text("not an image", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_rgb_image(bad_ext)

            corrupt_img = tmp_path / "corrupt.jpg"
            corrupt_img.write_bytes(b"not-valid-jpeg-bytes")
            with self.assertRaises(ValueError):
                load_rgb_image(corrupt_img)


class Set10ClassificationAndCachingTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_clip_cache()

    def test_zero_shot_classification_ranks_descending_and_sums_to_one(self) -> None:
        fake_model = _FakeCLIPModel()
        fake_proc = _FakeCLIPProcessor()
        img = Image.new("RGB", (32, 32), (10, 20, 30))

        res = classify_image_zero_shot(
            image_input=img,
            class_entries=[
                ("cat", ""),
                ("dog", "a photo of a golden retriever"),
                ("car", ""),
            ],
            model_name=DEFAULT_CLIP_MODEL,
            model_bundle=(fake_model, fake_proc, "cpu"),
        )

        self.assertTrue(fake_model.eval_called)
        self.assertEqual(res["top_label"], "dog")
        ranked = res["results"]
        self.assertEqual([r["label"] for r in ranked], ["dog", "cat", "car"])
        self.assertEqual(ranked[0]["prompt"], "a photo of a golden retriever")
        self.assertEqual(ranked[1]["prompt"], "a photo of a cat.")
        self.assertAlmostEqual(sum(r["probability"] for r in ranked), 1.0, places=5)

        rendered = render_clip_zero_shot_plot(img, res)
        self.assertEqual(rendered.mode, "RGB")
        self.assertGreater(rendered.width, 100)

    def test_load_clip_model_caching_and_failure_message(self) -> None:
        fake_model = _FakeCLIPModel()
        fake_proc = _FakeCLIPProcessor()

        with patch("forensics_app.tools.clip_core.CLIPProcessor.from_pretrained", return_value=fake_proc), \
             patch("forensics_app.tools.clip_core.CLIPModel.from_pretrained", return_value=fake_model):
            m1, p1, dev = load_clip_model("openai/clip-vit-base-patch32", device="cpu")
            self.assertIs(m1, fake_model)
            self.assertIs(p1, fake_proc)
            self.assertIsNotNone(get_cached_clip_model("openai/clip-vit-base-patch32", device=dev))

        clear_clip_cache()
        with patch(
            "forensics_app.tools.clip_core.CLIPProcessor.from_pretrained",
            side_effect=OSError("offline"),
        ):
            with self.assertRaises(CLIPModelLoadError) as ctx:
                load_clip_model("openai/clip-vit-base-patch32", device="cpu")
            self.assertIn("Could not load CLIP model", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

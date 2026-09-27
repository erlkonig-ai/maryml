"""CPU-only guards for the pinned Nomic adapter's namespace conversion."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multimodal_baseline_reference import nomic_adapter_key


class AdapterNamespaceTests(unittest.TestCase):
    def test_layer_identity_and_matrix_are_preserved(self):
        for layer in (0, 9, 27):
            for projection, block in (("q_proj", "self_attn"), ("down_proj", "mlp")):
                for side in ("A", "B"):
                    suffix = f"layers.{layer}.{block}.{projection}.lora_{side}"
                    self.assertEqual(
                        nomic_adapter_key(f"base_model.model.model.{suffix}.weight"),
                        f"language_model.{suffix}.default.weight")

    def test_unknown_or_already_converted_keys_fail_closed(self):
        for key in ("model.layers.0.self_attn.q_proj.lora_A.weight",
                    "base_model.model.visual.foo.weight",
                    "language_model.layers.0.self_attn.q_proj.lora_A.default.weight",
                    "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                nomic_adapter_key(key)


if __name__ == "__main__":
    unittest.main()

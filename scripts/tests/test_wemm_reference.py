"""CPU-only manifest guards for the offline BF16 reference harness."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("wemm_reference", Path(__file__).parents[1] / "wemm_reference.py")
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "note.txt").write_text("A local note.")
        self.item = {"id": "note", "group": "one", "split": "calibration", "role": "document",
                     "modality": "text", "path": "note.txt",
                     "sha256": hashlib.sha256(b"A local note.").hexdigest()}

    def read(self, items):
        path = self.root / "manifest.json"
        path.write_text(json.dumps({"items": items}))
        return REFERENCE.read_probe(path, self.root)

    def test_verified_text_and_hash(self):
        self.assertEqual(self.read([self.item])["items"][0]["text"], "A local note.")
        (self.root / "note.txt").write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.read([self.item])

    def test_traversal_fails_before_read(self):
        self.item["path"] = "../elsewhere"
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.read([self.item])

    def test_duplicate_and_split_leakage(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.read([self.item, copy.deepcopy(self.item)])
        other = {**self.item, "id": "other", "split": "heldout"}
        with self.assertRaisesRegex(ValueError, "spans"):
            self.read([self.item, other])

    def test_empty_text_and_image_without_path(self):
        item = {k: v for k, v in self.item.items() if k not in ("path", "sha256")}
        with self.assertRaisesRegex(ValueError, "empty"):
            self.read([{**item, "text": "  "}])
        with self.assertRaisesRegex(ValueError, "local path"):
            self.read([{**item, "modality": "image"}])

    def test_loader_sets_serialize_without_partial_output(self):
        path = self.root / "result.json"
        REFERENCE.write_report({"loading": {"missing_keys": set(), "unexpected_keys": {"b", "a"}}}, path)
        self.assertEqual(json.loads(path.read_text())["loading"]["unexpected_keys"], ["a", "b"])
        with self.assertRaises(FileExistsError):
            REFERENCE.write_report({}, path)
        bad_path = self.root / "bad.json"
        with self.assertRaises(TypeError):
            REFERENCE.write_report({"bad": object()}, bad_path)
        self.assertFalse(bad_path.exists())


if __name__ == "__main__":
    unittest.main()

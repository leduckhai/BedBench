"""The first two generated scenes must match the released scene pool exactly."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from bed_generator.scenes import generate

HERE = Path(__file__).parent


class ReproductionTest(unittest.TestCase):
    def test_first_scenes_match_released_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            generate(tmp, n_scenes=2, seed=42)
            for i in range(2):
                actual = json.loads(Path(tmp, f"sample_{i:04d}", "scene.json").read_text())
                expected = json.loads((HERE / f"reference_scene_{i:04d}.json").read_text())
                self.assertEqual(actual, expected)
            self.assertTrue(os.path.exists(os.path.join(tmp, "sample_0001/neg_uncolored/zx.png")))


if __name__ == "__main__":
    unittest.main()

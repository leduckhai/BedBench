"""The first two generated samples must match the released manifest exactly."""
import csv
import os
import tempfile
import unittest
from pathlib import Path

from bed_generator import generate

REFERENCE = Path(__file__).with_name("reference_head.csv")


class ReproductionTest(unittest.TestCase):
    def test_first_samples_match_released_split(self):
        with REFERENCE.open(newline="") as f:
            expected = list(csv.DictReader(f))
        with tempfile.TemporaryDirectory() as tmp:
            with open(generate(tmp, n_samples=len(expected), seed=42), newline="") as f:
                actual = list(csv.DictReader(f))
            self.assertTrue(os.path.exists(os.path.join(tmp, "samples/sample_00000/query1.png")))
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()

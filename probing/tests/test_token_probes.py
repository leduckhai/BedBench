import importlib
import unittest
import torch


class TokenProbeTests(unittest.TestCase):
    def test_neural_readouts_ignore_padding(self):
        probes = importlib.import_module("bed_probing.probe_tokens")
        torch.manual_seed(0)
        x = torch.randn(2, 5, 8)
        views = torch.tensor([[0, 0, 1, 1, 1]] * 2)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        for cls in (probes.AttnProbe, probes.TokenSumProbe):
            model = cls(8).eval()
            with torch.no_grad():
                expected = model(x, views, mask)
                actual = model(torch.cat([x, torch.randn(2, 3, 8)], 1),
                               torch.cat([views, torch.zeros(2, 3, dtype=torch.long)], 1),
                               torch.cat([mask, torch.ones(2, 3, dtype=torch.bool)], 1))
            self.assertEqual(expected.shape, (2, 12))
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual, expected)

    def test_token_training_returns_finite_test_counts(self):
        probes = importlib.import_module('bed_probing.probe_tokens')
        data = {}
        for split, n in [('train', 4), ('val', 2), ('test', 2)]:
            data[split] = (torch.randn(n, 5, 8), torch.zeros(n, 5, dtype=torch.long),
                           torch.zeros(n, 5, dtype=torch.bool), torch.ones(n, 12))
        for cls in (probes.AttnProbe, probes.TokenSumProbe):
            result = probes.train_neural(cls, data, seed=0, device='cpu', epochs=2, patience=1)
            self.assertEqual(result.shape, (2, 12))
            self.assertTrue(torch.isfinite(torch.as_tensor(result)).all())

    def test_connector_extracts_merged_tokens_not_tower_tokens(self):
        from types import SimpleNamespace
        extraction = importlib.import_module('bed_probing.extract_tokens')
        tower = extraction.TokenTower.__new__(extraction.TokenTower)
        tower.stage = 'connector'
        tower._out = SimpleNamespace(pooler_output=torch.ones(1, 3, 8),
                                     last_hidden_state=torch.zeros(1, 12, 4))
        tower.enc = SimpleNamespace(encode_one=lambda image: None)
        result = tower.tokens(None)
        self.assertEqual(result.shape, (3, 8))
        self.assertEqual(result.dtype, torch.float16)
        self.assertTrue((result == 1).all())
        tower._out = SimpleNamespace(last_hidden_state=torch.zeros(1, 12, 4))
        with self.assertRaises(RuntimeError):
            tower.tokens(None)

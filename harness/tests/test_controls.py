"""Offline execution checks: python -m unittest discover -s experiments/tests -v (from release root)."""
import asyncio
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


PROMPTS = Path(__file__).resolve().parents[2] / "prompts" / "non_thinking"


def module(folder, name):
    name = "build_count_inputs" if name == "build_inputs" else name
    return importlib.import_module("bed_eval." + name)


class ExecutionTests(unittest.TestCase):
    def test_counts_pair_by_object_id_and_reject_missing(self):
        build = module('t7_from_counts', 'build_inputs')
        rows = [{'row_id': x} for x in (1, 2)]
        with patch.object(build, 'load', side_effect=[rows, rows[::-1]]):
            pairs = build.paired(Path('unused'), 'model', 'color', 1)
        self.assertEqual([(a['row_id'], b['row_id']) for a, b in pairs], [(1, 1), (2, 2)])
        with patch.object(build, 'load', side_effect=[rows, rows[:1]]):
            with self.assertRaises(ValueError):
                build.paired(Path('unused'), 'model', 'color', 1)

    def test_counts_runner_uses_text_only_t7_payload(self):
        runner = module('t7_from_counts', 'run_t7_counts')
        captured = []

        async def post(session, endpoint, key, payload, retries):
            captured.append(payload)
            return {'choices': [{'message': {'content': '{"answer":["red"]}'}}]}

        rec = {'condition': 'self_count', 'variant': 'color', 'run': 1, 'row_id': 4,
               'counts': {'red': 2, 'blue': 1}, 'target': ['red']}
        args = SimpleNamespace(api_url='http://unused', model='model', concurrency=1)
        template = (PROMPTS / 'Task_7_counts.md').read_text()
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, '_post', post):
            out = Path(tmp) / 'run.jsonl'
            asyncio.run(runner.run_group([rec], template, out, args))
            saved = json.loads(out.read_text())
        self.assertFalse(saved['with_images'])
        self.assertEqual(saved['row_id'], 4)
        message = captured[0]['messages'][0]['content']
        self.assertEqual([part['type'] for part in message], ['text'])
        self.assertIn(json.dumps(rec['counts']), message[0]['text'])
        self.assertEqual(captured[0]['temperature'], 0)
        self.assertIn('response_format', captured[0])

    def test_prompt_runner_sends_both_variants(self):
        runner = module('t5_prompt_ablation', 'run_t5_prompt')
        calls = []

        async def evaluate(spec, rows, template, out, **kwargs):
            calls.append((template, out.name, kwargs))

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(runner, 'load_rows', return_value=[{}] * 100), \
                patch.object(runner, 'evaluate', evaluate), \
                patch('sys.argv', ['run', '--dataset', 'unused', '--model', 'model', '--runs', '1', '--out', tmp]):
            runner.main()
        self.assertEqual([c[1] for c in calls], ['color_no_example.jsonl', 'color_valid_example.jsonl'])
        self.assertNotIn('Example:', calls[0][0])
        self.assertIn('Example:', calls[1][0])
        self.assertTrue(all(not c[2]['thinking'] and c[2]['variant'] == 'color' for c in calls))

    def test_tile_runner_sends_one_image_and_unchanged_target(self):
        from PIL import Image
        runner = module('t5_2d_control', 'run_t5_2d')
        calls = []

        async def evaluate(spec, rows, template, out, **kwargs):
            calls.append((spec.images, rows, kwargs))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new('RGB', (4, 4), 'red').save(root / 'tile.png')
            manifest = root / 'rows.jsonl'
            manifest.write_text(''.join(json.dumps({'sample_id': i, 'image': 'tile.png',
                                'color_distribution_json': '{"red": 1}'}) + '\n' for i in range(100)))
            with patch.object(runner, 'evaluate', evaluate), \
                    patch('sys.argv', ['run', '--inputs', str(manifest), '--model', 'model',
                                       '--runs', '1', '--out', str(root / 'results')]):
                runner.main()
        self.assertEqual(calls[0][0], ('tiles',))
        self.assertEqual(len(calls[0][1]), 100)
        self.assertEqual(calls[0][1][0]['color_distribution_json'], '{"red": 1}')
        self.assertFalse(calls[0][2]['thinking'])



if __name__ == '__main__':
    unittest.main()

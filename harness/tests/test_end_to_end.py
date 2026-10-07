"""End-to-end check against a fake OpenAI-compatible server: payload, saved records, and scoring."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from aiohttp import web
from PIL import Image

from bed_eval.client import evaluate
from bed_eval.scoring import aggregate, score_run
from bed_eval.tasks import TASKS


class EndToEndTest(unittest.TestCase):
    def test_run_and_score(self):
        payloads = []

        async def handler(request):
            payload = await request.json()
            payloads.append(payload)
            answer = 3 if len(payloads) % 2 else 5  # one right, one wrong
            return web.json_response({"choices": [{"message": {"content": json.dumps({"answer": answer})}}]})

        async def scenario(out: Path):
            app = web.Application()
            app.router.add_post("/v1/chat/completions", handler)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            image = Image.new("RGB", (8, 8), "white")
            rows = [{"sample_id": str(i), "answer": 3, "query_image_1": image, "query_image_2": image} for i in range(2)]
            try:
                await evaluate(TASKS["T4"], rows, "Count the colors.", out, model="m", api_url=f"http://127.0.0.1:{port}",
                               variant="color", thinking=True, concurrency=1)
            finally:
                await runner.cleanup()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "r.jsonl"
            asyncio.run(scenario(out))
            records = [json.loads(line) for line in out.read_text().splitlines()]

        content = payloads[0]["messages"][0]["content"]
        self.assertEqual([c["type"] for c in content], ["image_url", "image_url", "text"])
        self.assertEqual((payloads[0]["temperature"], payloads[0]["top_k"], payloads[0]["thinking_token_budget"]), (0, 1, 1024))
        self.assertEqual(payloads[0]["response_format"]["json_schema"]["schema"]["required"], ["answer"])
        self.assertEqual([r["target"] for r in records], [3, 3])
        self.assertTrue(all(c["type"] == "text" for r in records for c in r["messages"][0]["content"]))
        self.assertEqual(aggregate([score_run(records)])["mean"]["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()

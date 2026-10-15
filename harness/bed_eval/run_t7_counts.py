"""Run text-only T7 from supplied counts using the released Non-Thinking payload. See README.md for usage."""
import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

import aiohttp


from .client import _post
from .requests import build_payload
from .tasks import TASKS

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts" / "non_thinking"


async def run_group(records: list[dict], template: str, out: Path, args) -> None:
    spec = TASKS["T7"]
    endpoint = args.api_url.rstrip("/") + "/v1/chat/completions"
    sem = asyncio.Semaphore(args.concurrency)

    async def one(rec: dict, session: aiohttp.ClientSession) -> dict:
        text = template.replace("{{counts}}", json.dumps(rec["counts"]))
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        payload = build_payload(spec, args.model, messages, thinking=False)
        async with sem:
            response = await _post(session, endpoint, "", payload, 5)
        content = ((response.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        return {**rec, "task": "T7", "model": args.model, "with_images": False, "messages": messages,
                "response": response, "content": content}

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
        results = await asyncio.gather(*(one(r, session) for r in records))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3.5-27B-T5-LoRA")
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", default="t7_counts_results")
    ap.add_argument("--concurrency", type=int, default=100)
    args = ap.parse_args()
    template = (PROMPT_DIR / "Task_7_counts.md").read_text()
    groups = defaultdict(list)
    for line in Path(args.inputs).read_text().splitlines():
        r = json.loads(line)
        groups[(r["condition"], r["variant"], r["run"])].append(r)
    for (cond, variant, run), records in sorted(groups.items()):
        out = Path(args.out) / cond / variant / f"run_{run}.jsonl"
        if out.exists():
            continue
        asyncio.run(run_group(records, template, out, args))
        print("wrote", out, len(records), flush=True)


if __name__ == "__main__":
    main()

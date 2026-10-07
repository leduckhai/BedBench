"""Asynchronous evaluation against an OpenAI-compatible endpoint; one JSONL record per question."""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import aiohttp

from .requests import build_messages, build_payload, text_only
from .tasks import TaskSpec

logger = logging.getLogger(__name__)
RETRY_STATUS = (502, 503)


def target_of(spec: TaskSpec, row: dict, fmt: str, color: Optional[str]):
    """Ground truth stored with each record, so scoring needs no dataset access."""
    if fmt == "single_color":
        return json.loads(row["color_distribution_json"]).get(color, 0)
    if spec.task == "T3":
        return {k: int(row[k]) for k in ("answer_intersection", "answer_only_a", "answer_only_b")}
    if spec.task == "T5":
        return json.loads(row["color_distribution_json"])
    if spec.task == "T12":
        return int(row["answer_idx"])
    return row["answer"]


async def _post(session: aiohttp.ClientSession, endpoint: str, api_key: str, payload: dict, retries: int) -> dict:
    for attempt in range(retries):
        try:
            async with session.post(endpoint, json=payload, headers={"Authorization": f"Bearer {api_key}"}) as resp:
                resp.raise_for_status()
                return await resp.json()
        except aiohttp.ClientResponseError as e:
            if e.status not in RETRY_STATUS or attempt == retries - 1:
                raise
            logger.warning("server returned %s; retrying in 30 s", e.status)
            await asyncio.sleep(30)
    raise RuntimeError("unreachable")


async def evaluate(spec: TaskSpec, rows: list[dict], template: str, out_path: Path, *, model: str, api_url: str,
                   api_key: str = "", variant: str, thinking: bool, fmt: str = "default", color: Optional[str] = None,
                   with_images: bool = True, thinking_budget: Optional[int] = 1024, concurrency: int = 100,
                   retries: int = 5) -> None:
    """Send every row, then write records in row order to out_path (JSONL)."""
    endpoint = api_url.rstrip("/")
    if not endpoint.endswith("/v1/chat/completions"):
        endpoint += "/v1/chat/completions"
    semaphore = asyncio.Semaphore(concurrency)

    async def one(i: int, row: dict, session: aiohttp.ClientSession) -> dict:
        messages = build_messages(spec, row, template, fmt=fmt, color=color, with_images=with_images)
        payload = build_payload(spec, model, messages, thinking=thinking, fmt=fmt, thinking_budget=thinking_budget)
        async with semaphore:
            response = await _post(session, endpoint, api_key, payload, retries)
        content = ((response.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        return {"idx": i, "row_id": row.get("sample_id", row.get("pair_id", i)), "task": spec.task,
                "variant": variant, "format": fmt, "color": color, "with_images": with_images,
                "thinking": thinking, "model": model, "target": target_of(spec, row, fmt, color),
                "messages": text_only(messages), "response": response, "content": content}

    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(*(one(i, row, session) for i, row in enumerate(rows)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("wrote %d records to %s", len(results), out_path)

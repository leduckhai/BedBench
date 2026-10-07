"""Chat-completion messages and payloads (OpenAI-compatible, vLLM extensions)."""
from __future__ import annotations

import base64
from io import BytesIO
from typing import Optional

from .tasks import TaskSpec, _INT, _schema, fill_prompt, image_legend

SINGLE_COLOR_SCHEMA = _schema("Task5SingleColorResponse", {"answer": _INT})


def image_url(image) -> str:
    """PNG bytes as a data URL. The original runs labelled them image/jpeg; servers decode by content."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def build_messages(spec: TaskSpec, row: dict, template: str, *, fmt: str = "default",
                   color: Optional[str] = None, with_images: bool = True) -> list[dict]:
    """One user turn: images and prompt in the task's order. with_images=False is the no-image control."""
    texts = [{"type": "text", "text": fill_prompt(spec, template, row, fmt, color)}]
    if spec.text_first:
        texts.append({"type": "text", "text": image_legend()})
    images = [{"type": "image_url", "image_url": {"url": image_url(row[c])}} for c in spec.images] if with_images else []
    return [{"role": "user", "content": texts + images if spec.text_first else images + texts}]


def build_payload(spec: TaskSpec, model: str, messages: list[dict], *, thinking: bool, fmt: str = "default",
                  thinking_budget: Optional[int] = 1024, skip_special_tokens: bool = False) -> dict:
    """Greedy decoding with a JSON-schema answer. thinking_budget=None omits the budget field (e.g. Kimi-VL)."""
    single = fmt == "single_color"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0,
        "top_k": 1,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": f"{spec.task}_single_color_response" if single else f"{spec.task}_response",
            "schema": SINGLE_COLOR_SCHEMA if single else spec.schema}},
        "chat_template_kwargs": {"enable_thinking": thinking},
        "max_tokens": 4096 if single else spec.max_tokens,
    }
    if thinking:
        if thinking_budget is not None:
            payload["thinking_token_budget"] = thinking_budget
        payload["skip_special_tokens"] = skip_special_tokens
    return payload


def text_only(messages: list[dict]) -> list[dict]:
    """Messages as stored with each result: image payloads dropped."""
    return [{"role": m["role"], "content": [c for c in m["content"] if c["type"] == "text"]} for m in messages]

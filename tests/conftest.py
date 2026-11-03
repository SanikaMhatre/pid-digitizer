"""Shared test helpers: a scripted fake chat client and a mock HTTP transport."""
import json

import httpx
import pytest

from pid.models.llm import ChatClient, ChatResult, Ledger


class ScriptedClient:
    """Stands in for ChatClient: returns pre-written ChatResults in order and records the calls."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.ledger = Ledger()

    def chat(self, messages, **kw):
        self.calls.append({"messages": [dict(m) for m in messages], **kw})
        if not self.replies:
            return ChatResult(text="", tool_calls=[], message={"role": "assistant", "content": ""})
        r = self.replies.pop(0)
        return r(messages) if callable(r) else r


def text_reply(text):
    return ChatResult(text=text, tool_calls=[], message={"role": "assistant", "content": text})


def tool_reply(*calls):
    tcs = [{"id": f"c{i}", "name": n, "arguments": a} for i, (n, a) in enumerate(calls)]
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": t["id"], "type": "function", "function": {"name": t["name"], "arguments": json.dumps(t["arguments"])}}
            for t in tcs
        ],
    }
    return ChatResult(text="", tool_calls=tcs, message=msg)


def openai_body(content=None, tool_calls=None, prompt_tokens=100, completion_tokens=20):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "id": "x",
        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


@pytest.fixture
def make_client(tmp_path):
    def _make(handler, cache=True, **kw):
        return ChatClient(
            base_url="https://example.test/v1",
            api_key="k",
            model="m",
            cache_dir=(tmp_path / "cache") if cache else False,
            transport=httpx.MockTransport(handler),
            sleep=lambda s: None,
            **kw,
        )

    return _make

"""Bounded, logged retries for OpenAI-compatible endpoints (no network)."""

from __future__ import annotations

import asyncio

import httpx

from dlens.agents._models import _RetryingTransport


def _send(statuses: list[int], headers: dict | None = None, **kw) -> tuple[int, int]:
    """Replay `statuses` in order; return (final status, requests made)."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(statuses[len(seen) - 1], headers=headers or {})

    t = _RetryingTransport(base_delay=0.0, inner=httpx.MockTransport(handler), **kw)

    async def go():
        async with httpx.AsyncClient(transport=t) as c:
            return await c.post("https://api.groq.com/openai/v1/chat/completions", json={})

    return asyncio.run(go()).status_code, len(seen)


def test_retries_429_and_5xx_then_succeeds(capsys):
    assert _send([429, 503, 200]) == (200, 3)
    log = capsys.readouterr().out
    assert log.count("[llm retry]") == 2 and "attempt 1/6" in log and "attempt 2/6" in log


def test_gives_up_loudly_after_six_attempts(capsys):
    assert _send([503] * 6) == (503, 6)  # the error response reaches the SDK, which raises
    assert "giving up after 6 attempts" in capsys.readouterr().out


def test_does_not_retry_client_errors(capsys):
    assert _send([400]) == (400, 1)
    assert "[llm retry]" not in capsys.readouterr().out


def test_retry_after_beyond_cap_gives_up_instead_of_waiting(capsys):
    assert _send([429, 200], headers={"retry-after": "3600"}) == (429, 1)
    assert "exceeds the 60s cap" in capsys.readouterr().out

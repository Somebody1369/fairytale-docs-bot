from types import SimpleNamespace

from bot.llm import LLM, parse_json


def test_json_in_code_block_or_with_words_around():
    assert parse_json('```json\n{"found": true}\n```') == {"found": True}
    assert parse_json('Ось відповідь: {"ids": [1, 2]} готово') == {"ids": [1, 2]}


async def test_simple_choices_go_to_the_lightest_model_first():
    llm = LLM("x", ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"])
    used = []

    async def create(model, **options):
        used.append((model, options.get("max_completion_tokens")))
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content='{"ids": []}'))])

    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    await llm.json("system", "user", max_tokens=700, fast=True)
    await llm.json("system", "user", max_tokens=1500)
    assert used == [("openai/gpt-oss-20b", 700), ("openai/gpt-oss-120b", 1500)]


def test_reply_that_is_not_an_object_is_not_json():
    import json
    import pytest
    with pytest.raises(json.JSONDecodeError):
        parse_json('[{"found": true}]')


async def test_model_out_of_its_daily_limit_rests():
    import groq
    import httpx

    llm = LLM("x", ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"])
    used = []

    async def create(model, **options):
        used.append(model)
        if model == "openai/gpt-oss-120b":
            response = httpx.Response(429, headers={"retry-after": "3000"}, request=httpx.Request("POST", "https://x"))
            raise groq.RateLimitError("tokens per day", response=response, body=None)
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content='{"a": 1}'))])

    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert await llm.json("system", "user") == {"a": 1}
    assert await llm.json("system", "user") == {"a": 1}
    assert used == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "qwen/qwen3.8-27b"]


def test_wait_comes_from_retry_after_or_the_token_reset():
    import groq
    import httpx

    from bot.llm import _retry_after

    def limited(**headers):
        response = httpx.Response(429, headers=headers, request=httpx.Request("POST", "https://x"))
        return groq.RateLimitError("limit", response=response, body=None)

    assert _retry_after(limited(**{"retry-after": "12"})) == 12
    assert _retry_after(limited(**{"x-ratelimit-reset-tokens": "7.5s"})) == 7.5
    assert _retry_after(limited(**{"x-ratelimit-reset-tokens": "1m2.5s"})) == 62.5
    assert _retry_after(limited()) is None

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


def rate_limited(message: str, **headers):
    import groq
    import httpx

    response = httpx.Response(429, headers=headers, request=httpx.Request("POST", "https://x"))
    return groq.RateLimitError(message, response=response, body=None)


def fake_client(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


OK = SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content='{"a": 1}'))])


def test_default_models_answer_with_gpt_oss_first_and_pick_with_the_smallest():
    from bot.config import DEFAULT_GROQ_MODELS

    llm = LLM("x", DEFAULT_GROQ_MODELS.split(","))
    assert llm.order(fast=False) == ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
    assert llm.order(fast=True) == ["openai/gpt-oss-20b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"]
    assert llm.client.max_retries == 0  # waiting is decided by LLM.json alone, not by the SDK as well


async def test_qwen_is_asked_for_no_more_output_than_its_minute_allows():
    llm = LLM("x", ["qwen/qwen3.8-27b"])
    used = []

    async def create(model, **options):
        used.append(options["max_completion_tokens"])
        return OK

    llm.client = fake_client(create)
    await llm.json("system", "user", max_tokens=2000)
    await llm.json("system", "user", max_tokens=700, fast=True)
    assert used == [900, 700]


async def test_request_above_the_output_limit_is_asked_again_smaller_without_waiting():
    import asyncio

    llm = LLM("x", ["vendor/model-9b", "openai/gpt-oss-20b"])
    used = []

    async def create(model, **options):
        used.append((model, options["max_completion_tokens"]))
        if model == "vendor/model-9b" and options["max_completion_tokens"] > 600:
            raise rate_limited("Error code: 429 - {'error': {'message': 'Request too large for model `vendor/model-9b` "
                               "on output tokens per minute (OTPM): Limit 600, Requested 1500.'}}", **{"retry-after": "20"})
        return OK

    llm.client = fake_client(create)
    async with asyncio.timeout(2):  # waiting the 20 s of retry-after would fail here
        assert await llm.json("system", "user", max_tokens=1500) == {"a": 1}
        assert await llm.json("system", "user", max_tokens=1500) == {"a": 1}
    assert used == [("vendor/model-9b", 1500), ("vendor/model-9b", 500), ("vendor/model-9b", 500)]


async def test_prompt_too_large_for_a_model_goes_to_the_next_one_at_once():
    import asyncio

    llm = LLM("x", ["openai/gpt-oss-120b", "openai/gpt-oss-20b"])
    used = []

    async def create(model, **options):
        used.append(model)
        if model == "openai/gpt-oss-120b":
            raise rate_limited("Request too large for model `openai/gpt-oss-120b` on tokens per minute (TPM): "
                               "Limit 8000, Requested 9500.", **{"retry-after": "20"})
        return OK

    llm.client = fake_client(create)
    async with asyncio.timeout(2):
        assert await llm.json("system", "user") == {"a": 1}
        assert await llm.json("system", "user") == {"a": 1}
    # A smaller prompt may still fit the strong model: it is not sent to rest.
    assert used == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"] * 2


async def test_wait_that_does_not_fit_the_deadline_goes_to_the_next_model():
    import asyncio
    import time

    llm = LLM("x", ["openai/gpt-oss-120b", "openai/gpt-oss-20b"])
    used = []

    async def create(model, **options):
        used.append(model)
        if model == "openai/gpt-oss-120b":
            raise rate_limited("Rate limit reached: tokens per minute", **{"retry-after": "10"})
        return OK

    llm.client = fake_client(create)
    async with asyncio.timeout(2):  # without the deadline the bot would wait the 10 s
        assert await llm.json("system", "user", deadline=time.monotonic() + 8) == {"a": 1}
    assert used == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]


async def test_no_reply_before_the_deadline_is_unavailable(monkeypatch):
    import asyncio
    import time

    import pytest

    from bot import llm as llm_module
    from bot.llm import LLMUnavailable

    monkeypatch.setattr(llm_module, "MIN_ATTEMPT", 0.05)
    llm = LLM("x", ["openai/gpt-oss-120b", "openai/gpt-oss-20b"])
    used = []

    async def create(model, **options):
        used.append(model)
        await asyncio.sleep(5)
        return OK

    llm.client = fake_client(create)
    started = time.monotonic()
    with pytest.raises(LLMUnavailable):
        await llm.json("system", "user", deadline=started + 0.3)
    assert time.monotonic() - started < 1 and used == ["openai/gpt-oss-120b"]

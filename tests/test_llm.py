import asyncio
import json
import time
from types import SimpleNamespace

import groq
import httpx
import pytest

from bot import llm as llm_module
from bot.config import DEFAULT_GROQ_MODELS
from bot.llm import LLM, LLMUnavailable, _retry_after, model_options, parse_json


def rate_limited(message: str, **headers):
    response = httpx.Response(429, headers=headers, request=httpx.Request("POST", "https://x"))
    return groq.RateLimitError(message, response=response, body=None)


def too_large(message: str):
    """Groq answers a request over a per-minute limit with 413, which the SDK raises as a plain APIStatusError."""
    response = httpx.Response(413, request=httpx.Request("POST", "https://x"))
    return groq.APIStatusError(f"Error code: 413 - {{'error': {{'message': '{message}'}}}}", response=response,
                               body=None)


def cut_off():
    """Groq's 400 for a reply that ran out of max_completion_tokens before its JSON was complete."""
    response = httpx.Response(400, request=httpx.Request("POST", "https://x"))
    return groq.BadRequestError("json_validate_failed: max completion tokens reached before generating a valid "
                                "document", response=response, body=None)


def fake_client(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


OK = SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content='{"a": 1}'))])


def test_json_in_code_block_or_with_words_around():
    assert parse_json('```json\n{"found": true}\n```') == {"found": True}
    assert parse_json('Ось відповідь: {"ids": [1, 2]} готово') == {"ids": [1, 2]}


async def test_simple_choices_go_to_the_lightest_model_first():
    llm = LLM("x", ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"])
    used = []

    async def create(model, **options):
        used.append((model, options.get("max_completion_tokens")))
        return OK

    llm.client = fake_client(create)
    await llm.json("system", "user", max_tokens=700, fast=True)
    await llm.json("system", "user", max_tokens=1500)
    assert used == [("openai/gpt-oss-20b", 700), ("openai/gpt-oss-120b", 1500)]


def test_reply_that_is_not_an_object_is_not_json():
    with pytest.raises(json.JSONDecodeError):
        parse_json('[{"found": true}]')


async def test_model_out_of_its_daily_limit_rests():
    llm = LLM("x", ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"])
    used = []

    async def create(model, **options):
        used.append(model)
        if model == "openai/gpt-oss-120b":
            raise rate_limited("tokens per day", **{"retry-after": "3000"})
        return OK

    llm.client = fake_client(create)
    assert await llm.json("system", "user") == {"a": 1}
    assert await llm.json("system", "user") == {"a": 1}
    assert used == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "qwen/qwen3.8-27b"]


async def test_free_in_says_how_long_every_model_still_rests():
    llm = LLM("x", ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"])
    calls = {"openai/gpt-oss-120b": "3000", "qwen/qwen3.8-27b": "1200"}

    async def create(model, **options):
        raise rate_limited("tokens per day", **{"retry-after": calls[model]})

    llm.client = fake_client(create)
    assert llm.free_in() is None  # nothing has rested yet
    with pytest.raises(LLMUnavailable):
        await llm.json("system", "user", deadline=time.monotonic() + 5)
    assert 1190 < llm.free_in() <= 1200  # the first to free up, as Groq said
    llm.resting["qwen/qwen3.8-27b"] = time.monotonic() - 1  # one model free again: no matter of limits
    assert llm.free_in() is None


def test_wait_comes_from_retry_after_or_the_token_reset():
    assert _retry_after(rate_limited("limit", **{"retry-after": "12"})) == 12
    assert _retry_after(rate_limited("limit", **{"x-ratelimit-reset-tokens": "7.5s"})) == 7.5
    assert _retry_after(rate_limited("limit", **{"x-ratelimit-reset-tokens": "1m2.5s"})) == 62.5
    assert _retry_after(rate_limited("limit")) is None


def test_default_models_answer_with_gpt_oss_first_and_pick_with_the_smallest():
    llm = LLM("x", DEFAULT_GROQ_MODELS.split(","))
    assert llm.order(fast=False) == ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
    # The strong model's minute is kept for the answer that comes right after the pick.
    assert llm.order(fast=True) == ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
    assert LLM("x", ["openai/gpt-oss-120b"]).order(fast=True) == ["openai/gpt-oss-120b"]  # the only one still picks
    assert llm.client.max_retries == 0  # waiting is decided by LLM.json alone, not by the SDK as well


async def test_configured_models_stay_as_they_are_and_defaults_stand_in_only_for_none_offered():
    offered = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

    async def models_list():
        return SimpleNamespace(data=[SimpleNamespace(id=model) for model in offered])

    async def checked(models: list[str]) -> list[str]:
        llm = LLM("x", models)
        llm.client = SimpleNamespace(models=SimpleNamespace(list=models_list))
        await llm.check_models()
        return llm.models

    # The README's advice for the paid tier, where qwen costs several times more: it must not come back.
    assert await checked(["openai/gpt-oss-120b", "openai/gpt-oss-20b"]) == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    assert await checked(["vendor/gone"]) == offered  # no configured model is offered any more
    assert await checked(DEFAULT_GROQ_MODELS.split(",")) == DEFAULT_GROQ_MODELS.split(",")


async def test_key_groq_does_not_accept_stops_the_bot_at_start():
    # Otherwise every question looked like a busy model, and the bot apologised ten minutes later.
    request = httpx.Request("GET", "https://api.groq.com/openai/v1/models")
    llm = LLM("gsk_typo", ["openai/gpt-oss-120b"])

    async def failing(error):
        async def models_list():
            raise error
        llm.client = SimpleNamespace(models=SimpleNamespace(list=models_list))
        await llm.check_models()

    with pytest.raises(SystemExit, match="GROQ_API_KEY"):
        await failing(groq.AuthenticationError("Invalid API Key", response=httpx.Response(401, request=request),
                                               body=None))
    # Groq or the network failing says nothing about the key: the bot starts with the configured models.
    await failing(groq.InternalServerError("Service Unavailable", response=httpx.Response(503, request=request),
                                           body=None))
    await failing(groq.APIConnectionError(request=request))
    assert llm.models == ["openai/gpt-oss-120b"]


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


async def test_prompt_with_its_answer_budget_over_the_minute_is_asked_once_with_a_smaller_budget():
    llm = LLM("x", ["openai/gpt-oss-120b", "openai/gpt-oss-20b"])
    used = []

    async def create(model, messages, **options):
        used.append((model, options["max_completion_tokens"]))
        # The user message is the prompt's tokens here; Groq counts them with the reply's budget.
        requested = int(messages[1]["content"]) + options["max_completion_tokens"]
        if model == "openai/gpt-oss-120b" and requested > 8000:
            raise too_large("Request too large for model `openai/gpt-oss-120b` in organization `org_x` on tokens per "
                            f"minute (TPM): Limit 8000, Requested {requested}, please reduce your message size.")
        return OK

    llm.client = fake_client(create)
    async with asyncio.timeout(2):
        assert await llm.json("system", "7200", max_tokens=1500) == {"a": 1}
        # With 7700 tokens of prompt, less than 400 tokens are left for the answer: the next model is asked.
        assert await llm.json("system", "7700", max_tokens=1500) == {"a": 1}
    assert used == [("openai/gpt-oss-120b", 1500), ("openai/gpt-oss-120b", 700),
                    ("openai/gpt-oss-120b", 1500), ("openai/gpt-oss-20b", 1500)]


async def test_prompt_over_the_input_limit_or_still_too_large_goes_to_the_next_model():
    llm = LLM("x", ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"])
    used = []

    async def create(model, **options):
        used.append((model, options["max_completion_tokens"]))
        if model == "qwen/qwen3.8-27b":  # the prompt alone is over: a smaller budget does not help
            raise too_large("Request too large for model `qwen/qwen3.8-27b` on input tokens per minute (ITPM): "
                            "Limit 7000, Requested 7834, please reduce your message size.")
        if model == "openai/gpt-oss-120b":  # asked smaller once, and still too large
            raise too_large("Request too large for model `openai/gpt-oss-120b` on tokens per minute (TPM): "
                            "Limit 8000, Requested 8500, please reduce your message size.")
        return OK

    llm.client = fake_client(create)
    async with asyncio.timeout(2):
        assert await llm.json("system", "user", max_tokens=1500) == {"a": 1}
    assert used == [("qwen/qwen3.8-27b", 900), ("openai/gpt-oss-120b", 1500), ("openai/gpt-oss-120b", 900),
                    ("openai/gpt-oss-20b", 1500)]


async def test_no_request_starts_with_too_little_time_left_after_the_models_queue(monkeypatch):
    monkeypatch.setattr(llm_module, "MIN_ATTEMPT", 0.3)
    llm = LLM("x", ["openai/gpt-oss-120b"])
    used = []

    async def create(model, **options):
        used.append(model)
        return OK

    llm.client = fake_client(create)
    await llm.queues["openai/gpt-oss-120b"].acquire()  # another question's request to the same model
    asyncio.get_running_loop().call_later(0.4, llm.queues["openai/gpt-oss-120b"].release)
    with pytest.raises(LLMUnavailable):
        await llm.json("system", "user", deadline=time.monotonic() + 0.6)
    assert used == []


async def test_wait_that_does_not_fit_the_deadline_goes_to_the_next_model():
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


def test_qwen_answers_without_thinking():
    # Its hidden thinking used up all 900 output tokens the free tier allows, and no reply was left.
    assert model_options("qwen/qwen3.8-27b", "medium") == {"reasoning_effort": "none", "reasoning_format": "hidden"}
    assert model_options("openai/gpt-oss-120b", "medium") == {"reasoning_effort": "medium"}


async def test_with_every_model_busy_the_first_to_free_up_is_waited_for():
    llm = LLM("x", ["openai/gpt-oss-120b", "openai/gpt-oss-20b"])
    used = []

    async def create(model, **options):
        used.append(model)
        if model == "openai/gpt-oss-120b" and used.count(model) <= 2:
            raise rate_limited("Rate limit reached: tokens per minute", **{"retry-after": "0.2"})
        if model == "openai/gpt-oss-20b":
            raise rate_limited("Rate limit reached: tokens per minute", **{"retry-after": "30"})
        return OK

    llm.client = fake_client(create)
    started = time.monotonic()
    assert await llm.json("system", "user", deadline=started + 10) == {"a": 1}
    # 120b: waited out once, limited again; 20b: limited for longer; then the round is over and 120b,
    # free first, is asked again instead of giving up.
    assert used == ["openai/gpt-oss-120b", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    assert time.monotonic() - started < 2 and llm.resting["openai/gpt-oss-20b"] > time.monotonic()


async def test_simple_choice_does_not_wait_for_the_model_it_never_asks():
    llm = LLM("x", ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"])
    used = []

    async def create(model, **options):
        used.append(model)
        return OK

    llm.client = fake_client(create)
    started = time.monotonic()
    # The strong model frees up first, but a title pick leaves it to the answer; the others rest for a minute.
    llm.resting = {"openai/gpt-oss-120b": started + 0.5, "openai/gpt-oss-20b": started + 60,
                   "qwen/qwen3.8-27b": started + 60}
    with pytest.raises(LLMUnavailable):
        await llm.json("system", "user", fast=True, deadline=started + 10)
    assert time.monotonic() - started < 0.3 and used == []


async def test_reply_cut_off_by_long_thinking_is_asked_again_with_less_thinking():
    llm = LLM("x", ["openai/gpt-oss-20b"])
    efforts = []

    async def create(model, **options):
        efforts.append(options["reasoning_effort"])
        if len(efforts) == 1:
            raise cut_off()
        return OK

    llm.client = fake_client(create)
    assert await llm.json("system", "user", reasoning="medium") == {"a": 1}
    assert efforts == ["medium", "low"]


async def test_reply_cut_off_on_qwen_is_asked_again_as_plain_text_not_the_same_way():
    llm = LLM("x", ["qwen/qwen3.8-27b"])
    sent = []

    async def create(model, **options):
        sent.append(options)
        if len(sent) == 1:
            raise cut_off()
        return OK

    llm.client = fake_client(create)
    # Dates are asked with reasoning="medium", but qwen never thinks: thinking less would send the same request.
    assert await llm.json("system", "user", reasoning="medium") == {"a": 1}
    assert "response_format" in sent[0] and "response_format" not in sent[1]

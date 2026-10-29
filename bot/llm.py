"""Groq chat completions with JSON output and fallback between free-tier models.

The free tier allows about 8000 tokens a minute and 200 thousand a day per model, and Groq counts
the prompt plus `max_completion_tokens` when it admits a request. So budgets stay small, requests
to one model go one at a time, and a short rate-limit pause is waited out on the strong model
before a weaker one is tried. A model that has used up its day rests until Groq says it may go on."""

import asyncio
import json
import logging
import re
import time
from collections import defaultdict

import groq

from .config import DEFAULT_GROQ_MODELS

log = logging.getLogger(__name__)

MAX_RATE_LIMIT_WAIT = 25  # seconds; longer waits switch to the next model instead
MAX_REST = 3600  # seconds a rate-limited model is skipped at most, whatever Groq says


class LLMUnavailable(Exception):
    pass


def parse_json(text: str) -> dict:
    """A JSON object from a reply that may wrap it in a code block or add words around it."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        result = json.loads(match.group(0))
    if not isinstance(result, dict):  # a list or a bare string is no answer either
        raise json.JSONDecodeError("not a JSON object", text, 0)
    return result


def model_options(model: str, reasoning: str) -> dict:
    # Reasoning models think before answering: briefly for simple choices, longer for answers with dates.
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": reasoning}
    if model.startswith("qwen/"):  # its thinking is kept out of the reply, which must be pure JSON
        return {"reasoning_effort": reasoning, "reasoning_format": "hidden"}
    return {}


class LLM:
    def __init__(self, api_key: str, models: list[str]):
        self.client = groq.AsyncGroq(api_key=api_key, max_retries=1, timeout=60)
        self.models = list(models)
        self.queues: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.resting: dict[str, float] = {}  # model -> time.monotonic() when it may be asked again

    async def check_models(self) -> None:
        """Keep configured models this Groq key can use; add known free-tier ones as fallbacks."""
        try:
            available = {model.id for model in (await self.client.models.list()).data}
        except groq.APIError as error:
            log.warning("Could not list Groq models: %s", error)
            return
        usable = [m for m in self.models if m in available]
        usable += [m for m in DEFAULT_GROQ_MODELS.split(",") if m in available and m not in usable]
        if usable:
            self.models = usable
        else:
            log.error("None of the Groq models %s is available for this key", self.models)
        log.info("Groq models in use: %s", ", ".join(self.models))

    async def json(self, system: str, user: str, max_tokens: int = 1500, reasoning: str = "low",
                   fast: bool = False) -> dict:
        """`fast`: a simple choice (documents by title, a document's place) goes to the lightest model
        first, which also keeps the strong model's token budget for answers."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_error: Exception | None = None
        for model in reversed(self.models) if fast else self.models:
            if self.resting.get(model, 0) > time.monotonic():
                continue
            strict, waited = True, False
            while True:
                try:
                    async with self.queues[model]:
                        response = await self.client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=0.1,
                            max_completion_tokens=max_tokens,
                            **({"response_format": {"type": "json_object"}} if strict else {}),
                            **model_options(model, reasoning),
                        )
                    if response.usage:
                        log.debug("Groq %s: %s prompt + %s completion tokens", model,
                                  response.usage.prompt_tokens, response.usage.completion_tokens)
                    return parse_json(response.choices[0].message.content or "")
                except groq.RateLimitError as error:
                    last_error = error
                    wait = _retry_after(error)
                    if waited or wait is None or wait > MAX_RATE_LIMIT_WAIT:
                        if wait and wait > MAX_RATE_LIMIT_WAIT:  # the daily limit: no use asking again soon
                            self.resting[model] = time.monotonic() + min(wait, MAX_REST)
                        log.warning("Groq model %s is rate-limited (%s), trying the next one", model,
                                    f"{wait:.0f} s" if wait is not None else str(error)[:200])
                        break
                    waited = True
                    log.info("Groq model %s: token limit for this minute reached, waiting %.0f s", model, wait)
                    async with self.queues[model]:  # the others wait for the same limit anyway
                        await asyncio.sleep(wait)
                except groq.BadRequestError as error:
                    last_error = error
                    if strict and "json_validate_failed" in str(error):
                        strict = False  # ask again and read the JSON from plain text
                        continue
                    log.warning("Groq model %s failed: %s", model, error)
                    break
                except json.JSONDecodeError as error:
                    last_error = error
                    log.warning("Groq model %s returned text that is not JSON", model)
                    break
                except groq.APIError as error:
                    last_error = error
                    log.warning("Groq request failed on %s: %s", model, error)
                    break
        raise LLMUnavailable(str(last_error or "all models are resting after their rate limits"))


def _retry_after(error: groq.RateLimitError) -> float | None:
    """Seconds to wait: "retry-after", or else when the minute's token budget resets ("7.66s", "1m2.5s")."""
    headers = getattr(error.response, "headers", None) or {}
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        pass
    reset = re.fullmatch(r"(?:(\d+)m)?([\d.]+)s", headers.get("x-ratelimit-reset-tokens") or "")
    return int(reset.group(1) or 0) * 60 + float(reset.group(2)) if reset else None

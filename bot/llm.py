"""Groq chat completions with JSON output and fallback between free-tier models.

The free tier allows about 8000 tokens a minute and 200 thousand a day per model, and Groq counts
the prompt plus `max_completion_tokens` when it admits a request. So budgets stay small, requests
to one model go one at a time, and a short rate-limit pause is waited out on the strong model
before a weaker one is tried. A model that has used up its day rests until Groq says it may go on.
All waiting is decided here: the SDK does not retry on its own, and a question has a deadline."""

import asyncio
import json
import logging
import math
import re
import time
from collections import defaultdict

import groq

from .config import DEFAULT_GROQ_MODELS

log = logging.getLogger(__name__)

MAX_RATE_LIMIT_WAIT = 25  # seconds; longer waits switch to the next model instead
MAX_REST = 3600  # seconds a rate-limited model is skipped at most, whatever Groq says
REQUEST_TIMEOUT = 25  # seconds for one request; a slow model gives way to the next one
MIN_ATTEMPT = 3  # seconds a request needs at least: with less time to the deadline no model is asked
# Groq's free tier gives qwen 1000 output tokens a minute and refuses a request whose
# max_completion_tokens alone is above that ("Request too large"). Other caps are learned from that error.
OUTPUT_CAPS = {"qwen/qwen3.8-27b": 900}
CAP_MARGIN = 100  # tokens kept below a learned cap
MIN_OUTPUT = 400  # an answer needs at least this much; a model capped lower is skipped
TOO_LARGE = re.compile(r"Request too large.*?\((\w+)\):\s*Limit (\d+)", re.S)


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


def model_size(model: str) -> float:
    """Billions of parameters from the name: "openai/gpt-oss-20b" is 20, "qwen/qwen3.8-27b" 27, unknown last."""
    sizes = re.findall(r"(\d+(?:\.\d+)?)b\b", model.lower())
    return float(sizes[-1]) if sizes else math.inf


class LLM:
    def __init__(self, api_key: str, models: list[str]):
        self.client = groq.AsyncGroq(api_key=api_key, max_retries=0, timeout=REQUEST_TIMEOUT)
        self.models = list(models)
        self.queues: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.resting: dict[str, float] = {}  # model -> time.monotonic() when it may be asked again
        self.output_caps: dict[str, int] = dict(OUTPUT_CAPS)

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

    def order(self, fast: bool) -> list[str]:
        """Answers go down the configured list; simple choices start from the smallest model."""
        return sorted(self.models, key=model_size) if fast else self.models

    async def json(self, system: str, user: str, max_tokens: int = 1500, reasoning: str = "low",
                   fast: bool = False, deadline: float | None = None) -> dict:
        """`fast`: a simple choice (documents by title, a document's place) goes to the smallest model
        first, which also keeps the strong model's token budget for answers. `deadline`: the
        time.monotonic() by which the reply is needed; a rate-limit wait that does not fit moves on
        to the next model, and when the time is up LLMUnavailable is raised."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_error: Exception | None = None
        for model in self.order(fast):
            if self.resting.get(model, 0) > time.monotonic():
                continue
            strict, waited = True, False
            while True:
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left < MIN_ATTEMPT:
                    raise LLMUnavailable(f"no time left for the reply ({last_error or 'slow replies'})")
                limit = min(max_tokens, self.output_caps.get(model, max_tokens))
                if limit < MIN_OUTPUT <= max_tokens:
                    break  # its output cap is too small for this request
                try:
                    async with asyncio.timeout(left), self.queues[model]:
                        response = await self.client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=0.1,
                            max_completion_tokens=limit,
                            **({"response_format": {"type": "json_object"}} if strict else {}),
                            **model_options(model, reasoning),
                        )
                    if response.usage:
                        log.debug("Groq %s: %s prompt + %s completion tokens", model,
                                  response.usage.prompt_tokens, response.usage.completion_tokens)
                    return parse_json(response.choices[0].message.content or "")
                except TimeoutError:
                    raise LLMUnavailable(f"no reply before the deadline ({last_error or 'slow replies'})")
                except groq.RateLimitError as error:
                    last_error = error
                    too_large = TOO_LARGE.search(str(error))
                    if too_large:
                        # One request above a per-minute limit: waiting will not help. An output cap can be
                        # met with a smaller max_completion_tokens; a prompt too big for the model cannot.
                        cap = int(too_large.group(2)) - CAP_MARGIN
                        if too_large.group(1) == "OTPM" and cap < limit:
                            self.output_caps[model] = cap
                            log.info("Groq model %s takes at most %d output tokens, asking with that", model, cap)
                            continue
                        log.warning("Groq model %s cannot take this request (%s), trying the next one",
                                    model, too_large.group(0)[-60:])
                        break
                    wait = _retry_after(error)
                    fits = left is None or wait is None or wait <= left - MIN_ATTEMPT
                    if waited or wait is None or wait > MAX_RATE_LIMIT_WAIT or not fits:
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

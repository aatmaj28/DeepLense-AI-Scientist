from __future__ import annotations
import asyncio
import random

import httpx
from openai import AsyncOpenAI
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.litellm import LiteLLMProvider

_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class _RetryingTransport(httpx.AsyncBaseTransport):
    """Bounded retries on 429/5xx with exponential backoff (honouring Retry-After).

    Every retry is printed, so it lands in the run's tee'd log. When the attempts run
    out, or Retry-After exceeds the per-wait cap, the error response is returned as-is
    and the OpenAI SDK raises its usual typed error: exhaustion fails loudly.
    """

    def __init__(self, attempts: int = 6, base_delay: float = 2.0, max_delay: float = 60.0,
                 inner: httpx.AsyncBaseTransport | None = None):
        self.attempts, self.base_delay, self.max_delay = attempts, base_delay, max_delay
        self._inner = inner or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        for attempt in range(1, self.attempts + 1):
            response = await self._inner.handle_async_request(request)
            code = response.status_code
            if code not in _RETRY_STATUS:
                return response
            if attempt == self.attempts:
                print(f"[llm retry] HTTP {code} from {host}: giving up after "
                      f"{self.attempts} attempts", flush=True)
                return response
            try:
                retry_after = float(response.headers.get("retry-after", ""))
            except ValueError:
                retry_after = None
            if retry_after is not None and retry_after > self.max_delay:
                print(f"[llm retry] HTTP {code} from {host}: Retry-After {retry_after:.0f}s "
                      f"exceeds the {self.max_delay:.0f}s cap, giving up", flush=True)
                return response
            delay = (retry_after if retry_after is not None
                     else min(self.max_delay,
                              self.base_delay * 2 ** (attempt - 1) * random.uniform(0.8, 1.2)))
            await response.aclose()
            print(f"[llm retry] HTTP {code} from {host}: attempt {attempt}/{self.attempts} "
                  f"failed, retrying in {delay:.1f}s", flush=True)
            await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        await self._inner.aclose()


# Wrapper for OpenAI model (when you want to use OpenAI API)
# pydantic-ai v2 dropped the deprecated `OpenAIModel` alias; base on `OpenAIChatModel`.
# With base_url it targets an OpenAI-compatible endpoint (e.g. Groq) instead, using
# bounded, logged retries in place of the SDK's silent ones.
class OpenAIModel(OpenAIChatModel):
    def __init__(self, model_name: str = "gpt-4o-mini", *, base_url: str | None = None,
                 api_key: str | None = None):
        if base_url is None:
            super().__init__(model_name=model_name)
            return
        client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0,
                             http_client=httpx.AsyncClient(transport=_RetryingTransport()))
        super().__init__(model_name=model_name, provider=OpenAIProvider(openai_client=client))


# Wrapper for Google Gemini (Gemini API, native pydantic-ai provider). The key is
# read from GOOGLE_API_KEY, then GEMINI_API_KEY, unless passed explicitly.
# Structured output is not yet verified against a live model: docs/KNOWN_ISSUES.md §4.
class GeminiModel(GoogleModel):
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: str | None = None):
        super().__init__(model_name=model_name, provider=GoogleProvider(api_key=api_key))


# Wrapper for Ollama model
class OllamaModel(OpenAIChatModel):
    def __init__(self, model_name: str = "gpt-oss:20b", port: str = "11434"):
        super().__init__(
            model_name=model_name,
            provider=OllamaProvider(base_url=f"http://localhost:{port}/v1"),
        )


# Wrapper for Llama cpp model
class LlamaCppModel(OpenAIChatModel):
    def __init__(self, model_name: str = "gpt-oss:20b", port: str = "8080"):
        super().__init__(
            model_name=model_name,
            provider=OpenAIProvider(base_url=f"http://localhost:{port}/v1"),
        )


# Wrapper for Outlines Llama cpp model - When you want to strictly force output JSON format
# NOTE: pydantic-ai v2 removed the built-in Outlines integration
# (`pydantic_ai.models.outlines`). Kept for API stability; imports are lazy so the
# package still imports, and instantiation raises a clear error until Outlines is
# re-integrated.
class OutlinesLlamaCppModel:
    def __init__(
        self,
        repo_id: str = "ggml-org/gpt-oss-20b-GGUF",
        filename: str = "gpt-oss-20b-mxfp4.gguf",
        *,
        n_ctx: int = 32768,
        chat_format: str | None = None,
    ):
        try:
            from pydantic_ai.models.outlines import OutlinesModel
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on pydantic-ai version
            raise RuntimeError(
                "OutlinesLlamaCppModel is unavailable: pydantic-ai v2 removed the built-in "
                "Outlines integration (pydantic_ai.models.outlines)."
            ) from exc
        from llama_cpp import Llama

        self.model = OutlinesModel.from_llamacpp(
            Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_ctx=n_ctx,
                chat_format=chat_format,
                verbose=False,
            )
        )


# Wrapper for vLLM model [Work in progress]
class VLLMProvider(OpenAIProvider):
    """OpenAI-compatible provider for a local vLLM server."""

    def __init__(
        self, base_url: str = "http://localhost:8000/v1", api_key: str = "local"
    ):
        super().__init__(base_url=base_url, api_key=api_key)


class VLLMModel(OpenAIChatModel):
    def __init__(self, model_name: str = "gpt-oss:20b", port: str = "8000"):
        super().__init__(
            model_name=model_name,
            provider=VLLMProvider(base_url=f"http://localhost:{port}/v1"),
        )


# Wrapper for LiteLLM model [Work in progress]
class LiteLLMModel(OpenAIChatModel):
    def __init__(self, model_name: str = "local-ollama", port: str = "4000"):
        super().__init__(
            model_name=model_name,
            provider=LiteLLMProvider(
                api_base=f"http://127.0.0.1:{port}/v1",
                api_key="local",
            ),
        )

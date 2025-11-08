from __future__ import annotations

# llm_api.py
import os
import time
from openai import OpenAI, RateLimitError, APIError

# Lazily initialized at first call to avoid import-time env issues
openai_client = None  # type: ignore[assignment]


def call_llm_api(prompt, model, temperature, max_tokens, chat_completions=True):
    """
    Unified OpenAI caller for both evaluation and prompt runners.

    - Preserves the original signature and behavior (no current-time injection).
    - Retries on rate limits and transient API errors with linear backoff (1s,2s,3s,4s,5s).
    - Uses the module-level `openai_client` instance.

    Args:
      prompt (str): Full user prompt to send.
      model (str): OpenAI model name (e.g., "gpt-4o").
      temperature (float): Sampling temperature.
      max_tokens (int): Max tokens for the response.
      chat_completions (bool): If True, use Chat Completions; else use Completions.

    Returns:
      str: Model response text. Returns a short error string after exhausting retries.
    """
    # Lazy init OpenAI client so that callers can load .env before first use
    global openai_client
    if openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Attempt to load from .env if python-dotenv is available
            try:
                from dotenv import load_dotenv

                load_dotenv()
                api_key = os.getenv("OPENAI_API_KEY")
            except Exception:
                pass
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your environment or .env."
            )
        openai_client = OpenAI(api_key=api_key)

    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            if chat_completions:
                # Chat Completions API: concise system preamble (no date injection)
                response = openai_client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant. Respond as concisely as possible.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content or ""
            else:
                # Legacy Completions API
                response = openai_client.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    prompt=prompt,
                )
                return response.choices[0].text or ""

        except RateLimitError:
            # Linear backoff: 1s, 2s, 3s, ...
            sleep_s = 1 + attempt
            print(f"⚠️ Rate limit hit, sleeping {sleep_s} sec...")
            time.sleep(sleep_s)
        except APIError as e:
            # Treat transient server errors similarly
            sleep_s = 1 + attempt
            print(f"⚠️ API error: {e}. Sleeping {sleep_s} sec...")
            time.sleep(sleep_s)
        except Exception as e:
            # Non-transient: bubble up immediately
            raise e

    return "[ERROR] rate limit retry exceeded"

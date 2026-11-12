import base64
import time

import httpx

from ..core.config import GEMINI_API_URL, get_gemini_api_key
from ..exceptions import ConfigError, QuotaExceededError, VisionAPIError
from ..prompts import vision_prompt

VISION_PROMPT = vision_prompt()


def transcribe_page_with_gemini(
    image_bytes: bytes, page_num: int, max_retries: int = 3
) -> str:
    """
    Sends a page image to Gemini Flash Lite for structured Markdown transcription.
    Includes up to 3 automatic retries with exponential backoff for transient network/SSL drops.
    Terminates cleanly with human-readable error messages if API limits or keys fail.
    """
    api_key = get_gemini_api_key()
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": VISION_PROMPT},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": b64_image,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 4096,
        },
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    for attempt in range(max_retries):
        try:
            with httpx.Client(http2=True, timeout=60.0) as client:
                response = client.post(
                    GEMINI_API_URL,
                    json=payload,
                    headers=headers,
                )

            # 1. Handle Rate Limit / Quota Exceeded (429) - Immediate termination
            if response.status_code == 429:
                msg = f"API quota or rate limit exceeded (HTTP 429) on page {page_num}."
                print(
                    f"\n{'=' * 60}\n[GEMINI ERROR] {msg}\nIngestion stopped to protect your API account.\n{'=' * 60}\n"
                )
                raise QuotaExceededError(msg)

            # 2. Handle Authentication Error (400 / 401 / 403) - Immediate termination
            if response.status_code in (400, 401, 403):
                msg = f"Gemini authentication failed (HTTP {response.status_code}) on page {page_num}."
                print(
                    f"\n{'=' * 60}\n[GEMINI ERROR] {msg}\nPlease check that your GEMINI_API_KEY in .env is valid and active.\n{'=' * 60}\n"
                )
                raise ConfigError(msg)

            # 3. Handle Transient Server Errors (500, 502, 503, 504) - Retry
            if response.status_code in (500, 502, 503, 504):
                if attempt < max_retries - 1:
                    backoff = 2 * (attempt + 1)
                    print(
                        f"\n[GEMINI RETRY {attempt + 1}/{max_retries}] Server error (HTTP {response.status_code}) on page {page_num}. Retrying in {backoff}s..."
                    )
                    time.sleep(backoff)
                    continue
                else:
                    msg = f"Gemini request failed with status {response.status_code} on page {page_num} after {max_retries} attempts: {response.text[:200]}"
                    raise VisionAPIError(msg)

            # 4. Handle other HTTP Errors
            if response.status_code != 200:
                msg = f"Gemini request failed with status {response.status_code} on page {page_num}: {response.text[:200]}"
                print(f"\n{'=' * 60}\n[GEMINI ERROR] {msg}\n{'=' * 60}\n")
                raise VisionAPIError(msg)

            # 5. Extract Text
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                print(f"\n[WARNING] Gemini returned no content for page {page_num}.")
                return ""

            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            return text.strip()

        except (QuotaExceededError, ConfigError):
            raise
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            if attempt < max_retries - 1:
                backoff = 2 * (attempt + 1)
                print(
                    f"\n[GEMINI RETRY {attempt + 1}/{max_retries}] Connection issue on page {page_num} ({exc}). Retrying in {backoff}s..."
                )
                time.sleep(backoff)
            else:
                msg = f"Failed to connect to Gemini API on page {page_num} after {max_retries} attempts: {exc}"
                print(f"\n{'=' * 60}\n[NETWORK ERROR] {msg}\n{'=' * 60}\n")
                raise VisionAPIError(msg) from exc

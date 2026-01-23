"""
Image Generator Module.
Optimized for standardized Gemini client and Hugging Face.
Implements Google Gemini API rate limit best practices:
- RPM (Requests Per Minute) tracking
- RPD (Requests Per Day) tracking with midnight Pacific reset
- Timezone-aware quota management
"""

import os
import logging
import random
import re
import requests
import base64
import time
from typing import Optional, Tuple, Callable, Any, Set, List, Dict
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque, defaultdict
from threading import Lock
from .clients.gemini import GeminiClient as GeminiClient
from google.genai import types

logger = logging.getLogger(__name__)


# Error categorization for smart retry logic
class RetryableError(Exception):
    """Base class for errors that should be retried."""
    pass


class NonRetryableError(Exception):
    """Base class for errors that should NOT be retried."""
    pass


class GeminiRateLimiter:
    """
    Rate limiter for Gemini API based on official rate limit documentation.

    Implements:
    - RPM (Requests Per Minute) tracking
    - RPD (Requests Per Day) tracking with midnight Pacific reset
    - Per-model rate limit tracking
    - Token usage tracking (TPM)

    Per Google's documentation:
    https://ai.google.dev/gemini-api/docs/rate-limits
    """

    # Default rate limits (conservative defaults - will be adjusted based on API feedback)
    DEFAULT_RPM = 15  # Conservative RPM for free tier
    DEFAULT_RPD = 1500  # Conservative RPD for free tier

    # Pacific timezone for daily reset (as per Google's documentation)
    PACIFIC_OFFSET = -8  # UTC-8 (PST), UTC-7 (PDT handled dynamically)

    def __init__(self, rpm: int = None, rpd: int = None):
        """Initialize rate limiter with custom limits."""
        self._rpm_limit = rpm or self.DEFAULT_RPM
        self._rpd_limit = rpd or self.DEFAULT_RPD

        # Thread-safe locks
        self._lock = Lock()

        # Request tracking per minute (rolling window)
        self._request_times = deque()

        # Request tracking per day (resets at midnight Pacific)
        self._daily_count = 0
        self._daily_date = self._get_pacific_date()

        # Per-model tracking (for models with specific limits like image generation)
        self._model_requests: Dict[str, deque] = defaultdict(lambda: deque())

        # Rate limit state
        self._rate_limited_until = None

        logger.info(f"RateLimiter initialized: RPM={self._rpm_limit}, RPD={self._rpd_limit}")

    def _get_pacific_date(self) -> datetime.date:
        """Get current date in Pacific timezone."""
        from datetime import timezone, timedelta
        # Pacific Time: UTC-8 (PST) or UTC-7 (PDT)
        utc_now = datetime.now(timezone.utc)
        pacific_offset = timedelta(hours=self.PACIFIC_OFFSET)
        # Check if DST is active (simplified - between March and November)
        if utc_now.month > 3 and utc_now.month < 11:
            pacific_offset = timedelta(hours=-7)
        pacific_time = utc_now + pacific_offset
        return pacific_time.date()

    def _check_daily_reset(self):
        """Check and reset daily counter if it's a new day in Pacific timezone."""
        current_date = self._get_pacific_date()
        if current_date != self._daily_date:
            with self._lock:
                if current_date != self._daily_date:
                    logger.info(f"Daily quota reset: {self._daily_date} -> {current_date}")
                    self._daily_count = 0
                    self._daily_date = current_date

    def _cleanup_old_requests(self, request_deque: deque, window_seconds: int = 60):
        """Remove requests older than the time window."""
        cutoff_time = time.time() - window_seconds
        while request_deque and request_deque[0] < cutoff_time:
            request_deque.popleft()

    def can_make_request(self, model: str = None) -> Tuple[bool, Optional[str]]:
        """
        Check if a request can be made without exceeding rate limits.

        Returns:
            Tuple of (allowed, reason_if_not_allowed)
        """
        with self._lock:
            # Check if we're currently rate-limited
            if self._rate_limited_until:
                if time.time() < self._rate_limited_until:
                    wait_time = self._rate_limited_until - time.time()
                    return False, f"Rate limited, wait {wait_time:.1f}s"
                else:
                    # Rate limit period expired
                    self._rate_limited_until = None

            # Check daily reset
            self._check_daily_reset()

            # Clean up old requests from rolling windows
            self._cleanup_old_requests(self._request_times)

            # Check RPM limit
            if len(self._request_times) >= self._rpm_limit:
                oldest_request = self._request_times[0]
                wait_time = 60 - (time.time() - oldest_request)
                return False, f"RPM limit reached, wait {wait_time:.1f}s"

            # Check RPD limit
            if self._daily_count >= self._rpd_limit:
                return False, f"RPD limit reached ({self._daily_count}/{self._rpd_limit})"

            return True, None

    def record_request(self, model: str = None, success: bool = True):
        """Record a request attempt (call after making the request)."""
        with self._lock:
            if success:
                current_time = time.time()
                self._request_times.append(current_time)
                self._daily_count += 1

                if model:
                    self._model_requests[model].append(current_time)
                    # Clean up old requests for this model
                    self._cleanup_old_requests(self._model_requests[model])

    def record_rate_limit(self, retry_after: float = None):
        """Record that we were rate limited and should back off."""
        with self._lock:
            # Default backoff: 60 seconds, or use retry_after if provided
            backoff = retry_after or 60
            self._rate_limited_until = time.time() + backoff
            logger.warning(f"Rate limit detected, backing off for {backoff}s")

    def get_stats(self) -> Dict:
        """Get current rate limit statistics."""
        with self._lock:
            self._cleanup_old_requests(self._request_times)
            self._check_daily_reset()

            return {
                "rpm_current": len(self._request_times),
                "rpm_limit": self._rpm_limit,
                "rpd_current": self._daily_count,
                "rpd_limit": self._rpd_limit,
                "rpm_utilization": len(self._request_times) / self._rpm_limit * 100,
                "rpd_utilization": self._daily_count / self._rpd_limit * 100,
                "rate_limited_until": self._rate_limited_until,
            }

    def wait_if_needed(self, model: str = None) -> float:
        """
        Wait if rate limit would be exceeded. Returns wait time in seconds.

        This implements proactive throttling to avoid 429 errors.
        """
        can_request, reason = self.can_make_request(model)
        if not can_request:
            # Extract wait time from reason
            import re
            match = re.search(r'wait ([\d.]+)s', str(reason))
            wait_time = float(match.group(1)) if match else 5.0

            logger.info(f"Rate limit proactive throttle: {reason}")
            time.sleep(wait_time)
            return wait_time
        return 0.0


# Global rate limiter instance
_gemini_rate_limiter = GeminiRateLimiter()


def extract_http_status_code(error: Exception) -> Optional[int]:
    """Extract HTTP status code from various exception types."""
    error_str = str(error)

    # Check for Google API error format: "429 RESOURCE_EXHAUSTED"
    match = re.search(r"'code':\s*(\d+)", error_str)
    if match:
        return int(match.group(1))

    # Check for direct status code at start of error message
    match = re.search(r'^(\d{3})\s', error_str)
    if match:
        return int(match.group(1))

    # Check for status code in error string (e.g., "429 Too Many Requests")
    match = re.search(r'\b(\d{3})\b', error_str)
    if match:
        return int(match.group(1))

    return None


def call_with_smart_retry(
    func: Callable,
    service_name: str,
    max_retries: int = 8,
    base_delay: float = 1.0,
    max_delay: float = 120.0,
    retryable_statuses: Optional[Set[int]] = None,
    retryable_error_patterns: Optional[List[str]] = None,
    model: str = None,
    use_rate_limiter: bool = False
) -> Any:
    """
    Execute function with smart retry logic using truncated exponential backoff.

    Truncated Exponential Backoff:
        delay = min(base_delay * (2 ^ attempt) + jitter, max_delay)

    Args:
        func: Function to execute
        service_name: Name of the service (for logging)
        max_retries: Maximum number of retry attempts (default: 8)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay between retries in seconds (default: 120.0)
        retryable_statuses: HTTP status codes that are retryable
        retryable_error_patterns: List of regex patterns for retryable error messages
        model: Model name for per-model rate limiting
        use_rate_limiter: Whether to use the proactive rate limiter

    Returns:
        Function result or None if all retries fail
    """
    # Proactive rate limiting check (throttle before making the request)
    if use_rate_limiter and model and "gemini" in model.lower():
        _gemini_rate_limiter.wait_if_needed(model)
    if retryable_statuses is None:
        retryable_statuses = {429, 500, 502, 503, 504}

    if retryable_error_patterns is None:
        retryable_error_patterns = [
            r'RESOURCE_EXHAUSTED',
            r'QUOTA',
            r'RATE.?LIMIT',
            r'TIMEOUT',
            r'UNAVAILABLE',
            r'INTERNAL',
            r'temporary',
            r'transient'
        ]

    non_retryable_statuses = {400, 401, 403, 404, 422}
    non_retryable_patterns = [
        r'INVALID',
        r'NOT.?FOUND',
        r'PERMISSION',
        r'AUTHENTICAT',
        r'FORBIDDEN'
    ]

    for attempt in range(max_retries):
        try:
            result = func()
            if attempt > 0:
                logger.info(f"✅ {service_name} succeeded on attempt {attempt + 1}")

            # Record successful request in rate limiter
            if use_rate_limiter and model and "gemini" in model.lower():
                _gemini_rate_limiter.record_request(model, success=True)

            return result

        except Exception as e:
            error_msg = str(e)
            status_code = extract_http_status_code(e)

            # Record rate limit event for 429 errors
            if use_rate_limiter and status_code == 429 and model and "gemini" in model.lower():
                # Calculate suggested backoff time based on exponential backoff
                exponential_delay = base_delay * (2 ** attempt)
                _gemini_rate_limiter.record_rate_limit(retry_after=exponential_delay)

            # Check if error is non-retryable
            is_non_retryable = False

            if status_code in non_retryable_statuses:
                is_non_retryable = True
                logger.error(f"❌ {service_name}: Non-retryable HTTP {status_code} - {error_msg[:100]}")
                break

            for pattern in non_retryable_patterns:
                if re.search(pattern, error_msg, re.IGNORECASE):
                    is_non_retryable = True
                    logger.error(f"❌ {service_name}: Non-retryable error pattern '{pattern}' - {error_msg[:100]}")
                    break

            if is_non_retryable:
                break

            # Check if error is retryable
            is_retryable = False

            if status_code in retryable_statuses:
                is_retryable = True

            for pattern in retryable_error_patterns:
                if re.search(pattern, error_msg, re.IGNORECASE):
                    is_retryable = True
                    break

            if not is_retryable:
                logger.error(f"❌ {service_name}: Unclassified error (no retry) - {error_msg[:100]}")
                break

            # Calculate truncated exponential backoff with jitter
            # delay = base_delay * 2^attempt + jitter, capped at max_delay
            exponential_delay = base_delay * (2 ** attempt)
            jitter = random.uniform(0, min(exponential_delay * 0.1, 1.0))  # Up to 10% jitter, max 1s
            delay = min(exponential_delay + jitter, max_delay)

            # Log retry info with status code if available
            status_info = f"HTTP {status_code}" if status_code else "error"
            logger.warning(f"⚠️ {service_name}: {status_info} - Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)

    logger.error(f"❌ {service_name} failed after {max_retries} attempts")

    # Record failed request in rate limiter
    if use_rate_limiter and model and "gemini" in model.lower():
        _gemini_rate_limiter.record_request(model, success=False)

    return None

class ImageGenerator:
    """Generate featured images using various AI services."""

    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        self.gemini_client = gemini_client
        self.hf_token = os.environ.get("HUGGINGFACE_API_KEY")
        self.session = requests.Session()

    def generate_image(self, prompt: str, mode: str = "daily") -> Optional[bytes]:
        """Priority: Hugging Face FLUX.1 -> DALL-E 3 -> Gemini 3 Pro Image Preview."""

        # 1. Try Hugging Face FLUX.1-dev FIRST (Primary)
        if self.hf_token:
            try:
                # Enhance prompt for better quality
                enhanced_prompt = f"{prompt}, photorealistic, 8k, high fidelity, highly detailed, professional photography, cinematic lighting"
                logger.info(f"📸 Attempting Hugging Face FLUX.1 generation (Primary)...")
                hf_image = self.generate_image_huggingface(enhanced_prompt)
                if hf_image:
                    return hf_image
            except Exception as e:
                logger.warning(f"Hugging Face failed: {e}")

        # 2. Try DALL-E 3 (Fallback)
        dalle_image = self.generate_image_dalle(prompt)
        if dalle_image:
            return dalle_image

        # 3. Try Gemini 3 Pro Image Preview (Last Resort - requires API key, not Vertex AI)
        if self.gemini_client and self.gemini_client.client and not self.gemini_client.is_using_vertexai():
            logger.info("📸 Attempting Gemini 3 Pro Image Preview (fallback)...")

            # Show rate limiter stats
            stats = _gemini_rate_limiter.get_stats()
            logger.info(f"Rate Limit Stats: RPM {stats['rpm_current']}/{stats['rpm_limit']} "
                       f"({stats['rpm_utilization']:.1f}%), RPD {stats['rpd_current']}/{stats['rpd_limit']} "
                       f"({stats['rpd_utilization']:.1f}%)")

            def _try_gemini():
                config_options = types.GenerateContentConfig(
                    tools=[{"google_search": {}}],
                    image_config=types.ImageConfig(
                        aspect_ratio="1:1",
                        image_size="4K"
                    )
                )
                response = self.gemini_client.client.models.generate_content(
                    model="gemini-3-pro-image-preview",
                    contents=f"Generate a cinematic art piece based on: {prompt}",
                    config=config_options
                )
                if response.parts:
                    for part in response.parts:
                        if part.inline_data:
                            logger.info("✅ Image generated via Gemini 3 Pro Image Preview (4K)")
                            return part.inline_data.data
                raise ValueError("No image data in response")

            result = call_with_smart_retry(
                _try_gemini,
                service_name="Gemini 3 Pro Image Preview",
                max_retries=8,
                base_delay=1.0,
                max_delay=120.0,
                model="gemini-3-pro-image-preview",
                use_rate_limiter=True
            )
            if result:
                return result
        elif self.gemini_client and self.gemini_client.is_using_vertexai():
            logger.info("Vertex AI detected - Gemini 3 Pro Image requires Gemini API (API key)")
            logger.info("Skipping Gemini (not available)")
        else:
            logger.info("Gemini client not available")

        return None

    def generate_image_huggingface(self, prompt: str) -> Optional[bytes]:
        api_url = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-dev"
        headers = {"Authorization": f"Bearer {self.hf_token}"}

        def _try_hf():
            response = self.session.post(api_url, headers=headers, json={"inputs": prompt}, timeout=60)
            if response.status_code == 200:
                logger.info("✅ Image generated via Hugging Face")
                return response.content
            elif response.status_code == 429:
                # Rate limited - retry
                raise requests.HTTPError(f"Rate limited: {response.status_code}")
            elif response.status_code >= 500:
                # Server error - retry
                raise requests.HTTPError(f"Server error: {response.status_code}")
            else:
                # Client error - don't retry (raise to let smart retry handle categorization)
                raise requests.HTTPError(f"Client error {response.status_code}: {response.text[:100]}")

        return call_with_smart_retry(
            _try_hf,
            service_name="Hugging Face",
            max_retries=8,
            base_delay=1.0,
            max_delay=120.0
        )

    def generate_image_dalle(self, prompt: str) -> Optional[bytes]:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None

        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        def _try_dalle():
            logger.info("Generating with DALL-E 3...")
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1024x1024",
                response_format="b64_json"
            )
            return base64.b64decode(response.data[0].b64_json)

        return call_with_smart_retry(
            _try_dalle,
            service_name="DALL-E 3",
            max_retries=8,
            base_delay=1.0,
            max_delay=120.0
        )

    def save_image(self, image_data: bytes, filename: str) -> Optional[str]:
        try:
            images_dir = Path("generated_images")
            images_dir.mkdir(exist_ok=True)
            filepath = images_dir / filename
            with open(filepath, 'wb') as f:
                f.write(image_data)
            logger.info(f"Image saved to: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return None

class WordPressMediaUploader:
    """Upload images to WordPress media library."""

    def __init__(self, wp_url: str, wp_user: str, wp_app_password: str):
        self.wp_url = wp_url.rstrip('/')
        self.credentials = f"{wp_user}:{wp_app_password}"
        self.session = requests.Session()

    def upload_media(self, file_path: str, alt_text: str = "", title: str = "") -> Optional[int]:
        """Upload an image to WordPress and return the attachment ID."""
        url = f"{self.wp_url}/wp-json/wp/v2/media"
        token = base64.b64encode(self.credentials.encode()).decode('utf-8')
        headers = {
            "Authorization": f"Basic {token}",
            "Content-Type": "image/jpeg",
            "Content-Disposition": f"attachment; filename={os.path.basename(file_path)}"
        }

        try:
            with open(file_path, 'rb') as f:
                image_data = f.read()

            response = self.session.post(
                url,
                headers=headers,
                data=image_data,
                params={"alt_text": alt_text, "title": title},
                timeout=45
            )

            if response.status_code == 201:
                media_id = response.json().get('id')
                logger.info(f"✅ Media uploaded to WordPress (ID: {media_id})")
                return media_id
            else:
                logger.error(f"Media upload failed: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Media upload exception: {e}")
        return None

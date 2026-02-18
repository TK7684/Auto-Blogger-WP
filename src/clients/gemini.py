import logging
import os
import json
from pathlib import Path
from typing import Optional, Any, Dict
import httpx
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.genai.errors import ClientError

logger = logging.getLogger(__name__)


# OpenRouter Model Mapping
OPENROUTER_MODEL_MAP = {
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
    "gemini-2.0-flash-thinking-exp": "google/gemini-2.0-flash-thinking-exp:free",
    "gemini-1.5-pro": "google/gemini-pro-1.5",
    "gemini-1.5-flash": "google/gemini-flash-1.5",
    # Nanobana Image Generation Models (verified from OpenRouter API)
    "gemini-2.5-flash-image": "google/gemini-2.5-flash-preview-05-20",  # Updated model ID
    "gemini-3-pro-image": "google/gemini-3-pro-image-preview",
}


class GeminiClient:
    """Centralized client for Gemini API interactions.

    Supports:
    - Vertex AI with regional endpoint rotation
    - Google AI API (API key)
    - OpenRouter API (alternative backend)
    """

    # Available regions for Vertex AI (in order of preference)
    REGIONS = ["us-central1", "us-east1", "us-west1", "europe-west1", "asia-southeast1"]
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: Optional[str] = None, service_account_file: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.service_account_file = service_account_file or os.environ.get("GEMINI_SERVICE_ACCOUNT_KEY_FILE")
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        self._using_vertexai = False
        self._using_openrouter = False
        self._current_region_index = 0
        self._credentials = None
        self._project_id = None
        self.client = self._initialize_client()

    def _initialize_client(self) -> Optional[genai.Client]:
        """Initialize the GenAI client with preferred authentication.

        Priority:
        1. OpenRouter (if OPENROUTER_API_KEY is set)
        2. Vertex AI (if service account file exists)
        3. Google AI API (if GEMINI_API_KEY is set)
        """
        try:
            # 1. Check for OpenRouter first
            if self.openrouter_api_key:
                logger.info("Initializing Gemini with OpenRouter backend")
                self._using_openrouter = True
                return None  # No genai.Client needed for OpenRouter

            # 2. Check for Vertex AI service account
            if self.service_account_file:
                resolved_path = Path(self.service_account_file).resolve()
                if resolved_path.exists():
                    import google.oauth2.service_account as sa
                    logger.info(f"Initializing Gemini with service account (region rotation enabled): {resolved_path}")
                    self._credentials = sa.Credentials.from_service_account_file(
                        str(resolved_path),
                        scopes=['https://www.googleapis.com/auth/cloud-platform']
                    )
                    self._project_id = self._credentials.project_id
                    self._using_vertexai = True
                    return self._create_client_for_region(self.REGIONS[0])
                else:
                    logger.warning(f"Service account file not found: {resolved_path}")

            # 3. Fallback to API key
            if self.api_key:
                logger.info("Initializing Gemini with API key")
                self._using_vertexai = False
                return genai.Client(api_key=self.api_key)

            logger.error("No valid Gemini credentials found")
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {e}")
        return None

    def _create_client_for_region(self, region: str) -> genai.Client:
        """Create a Vertex AI client for a specific region."""
        return genai.Client(
            vertexai=True,
            project=self._project_id,
            location=region,
            credentials=self._credentials
        )

    def rotate_region(self) -> str:
        """Rotate to the next region and return the region name."""
        self._current_region_index = (self._current_region_index + 1) % len(self.REGIONS)
        new_region = self.REGIONS[self._current_region_index]
        logger.info(f"Rotating to region: {new_region}")
        self.client = self._create_client_for_region(new_region)
        return new_region

    def get_current_region(self) -> str:
        """Get the current region being used."""
        return self.REGIONS[self._current_region_index]

    def is_using_vertexai(self) -> bool:
        """Check if the client is using Vertex AI authentication."""
        return self._using_vertexai

    def is_using_openrouter(self) -> bool:
        """Check if the client is using OpenRouter backend."""
        return self._using_openrouter

    def _map_model_to_openrouter(self, model: str) -> str:
        """Map a Gemini model name to its OpenRouter equivalent."""
        return OPENROUTER_MODEL_MAP.get(model, f"google/{model}")

    def _generate_openrouter_content(self, model: str, contents: Any, config: Optional[types.GenerateContentConfig] = None) -> Any:
        """Generate content using OpenRouter API."""
        openrouter_model = self._map_model_to_openrouter(model)
        logger.info(f"Calling OpenRouter API (Model: {openrouter_model})")

        # Convert contents to OpenRouter chat format
        messages = []
        if isinstance(contents, str):
            messages = [{"role": "user", "content": contents}]
        elif isinstance(contents, list):
            for item in contents:
                if isinstance(item, str):
                    messages.append({"role": "user", "content": item})
                elif hasattr(item, 'text'):
                    messages.append({"role": "user", "content": item.text})
                elif isinstance(item, dict):
                    messages.append(item)
        else:
            messages = [{"role": "user", "content": str(contents)}]

        # Build request payload
        payload = {
            "model": openrouter_model,
            "messages": messages,
        }

        # Add config options if provided
        if config:
            if hasattr(config, 'temperature') and config.temperature is not None:
                payload["temperature"] = config.temperature
            # Prefer strict JSON-schema enforcement over generic json_object when schema is available
            if hasattr(config, 'response_json_schema') and config.response_json_schema is not None:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response_schema",
                        "strict": True,
                        "schema": config.response_json_schema,
                    },
                }
            elif hasattr(config, 'response_mime_type') and config.response_mime_type == "application/json":
                payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get("SITE_URL", "https://example.com"),
            "X-Title": os.environ.get("SITE_NAME", "Auto-Blogger"),
        }

        with httpx.Client(timeout=120.0) as http_client:
            response = http_client.post(
                f"{self.OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()

        # Create a response object that mimics genai response structure
        class OpenRouterResponse:
            def __init__(self, data: dict):
                self._data = data
                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                self.text = message.get("content", "")

            def __repr__(self):
                return f"OpenRouterResponse(text='{self.text[:50]}...')"

        return OpenRouterResponse(data)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((ClientError, httpx.HTTPStatusError)),
        reraise=True
    )
    def generate_content(self, model: str, contents: Any, config: Optional[types.GenerateContentConfig] = None) -> Any:
        """Generate content with retry logic for API errors."""
        if self._using_openrouter:
            return self._generate_openrouter_content(model, contents, config)

        if not self.client:
            raise RuntimeError("Gemini client not initialized")

        logger.info(f"Calling Gemini API (Model: {model})")
        return self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )

    def generate_structured_output(self, model: str, prompt: str, schema: Dict, tools: Optional[list] = None) -> Optional[Any]:
        """Generate content expected to match a specific JSON schema with model fallback."""
        # Primary model list with fallback options
        model_fallback_order = [model, "gemini-2.0-flash", "gemini-1.5-flash"]
        
        # Build config - avoid thinking_level for models that don't support it
        config_params = {
            "response_mime_type": "application/json",
            "response_json_schema": schema,
            "temperature": 1.0
        }
        
        # Only add tools if provided (avoid None value)
        if tools is not None:
            config_params["tools"] = tools
        
        # Avoid thinking_level parameter for models that don't support it (gemini-2.0-flash)
        # Models that support thinking_level: gemini-2.0-flash-thinking-exp, gemini-2.5-pro
        models_with_thinking = [
            "gemini-2.0-flash-thinking-exp",
            "gemini-2.5-pro",
            "gemini-2.5-pro-preview",
            "gemini-3-pro",
            "gemini-3-pro-preview"
        ]
        
        if any(thinking_model in model for thinking_model in models_with_thinking):
            config_params["thinking_level"] = "medium"
        
        config = types.GenerateContentConfig(**config_params)
        
        # Try each model in fallback order
        for attempt_model in model_fallback_order:
            try:
                logger.info(f"Attempting structured output with model: {attempt_model}")
                response = self.generate_content(attempt_model, prompt, config)
                if response and response.text:
                    logger.info(f"✅ Structured output generated with model: {attempt_model}")
                    return response
            except Exception as e:
                logger.warning(f"⚠️ Model {attempt_model} failed: {e}")
                continue
        
        logger.error(f"❌ All models failed for structured output generation")
        return None

    def generate_image_openrouter(
        self,
        prompt: str,
        model: str = "gemini-2.5-flash-image",
        size: str = "1024x1024"
    ) -> Optional[bytes]:
        """Generate an image using Nanobana (Gemini Image) via OpenRouter.

        Args:
            prompt: Text description of the image to generate
            model: Model to use (gemini-2.5-flash-image or gemini-3-pro-image)
            size: Image size (not directly used, for future compatibility)

        Returns:
            Image bytes if successful, None otherwise
        """
        import base64

        if not self._using_openrouter:
            logger.warning("generate_image_openrouter requires OpenRouter mode")
            return None

        openrouter_model = self._map_model_to_openrouter(model)
        logger.info(f"🎨 Generating image via Nanobana (Model: {openrouter_model})")

        # Build the image generation prompt
        image_prompt = f"Generate an image: {prompt}"

        payload = {
            "model": openrouter_model,
            "messages": [{"role": "user", "content": image_prompt}],
            "modalities": ["image", "text"],  # Required for image generation
        }

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get("SITE_URL", "https://example.com"),
            "X-Title": os.environ.get("SITE_NAME", "Auto-Blogger"),
        }

        try:
            with httpx.Client(timeout=180.0) as http_client:
                response = http_client.post(
                    f"{self.OPENROUTER_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()

            # Extract image from response
            choices = data.get("choices", [])
            if not choices:
                logger.error("No choices in OpenRouter response")
                return None

            message = choices[0].get("message", {})
            content = message.get("content", "")

            # Check if response contains base64 image data in different formats
            # Format 1: content is a list with image parts
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        # OpenRouter image_url format
                        if part.get("type") == "image_url":
                            image_url = part.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:image"):
                                base64_data = image_url.split(",", 1)[1]
                                logger.info("✅ Image generated via Nanobana (image_url format)")
                                return base64.b64decode(base64_data)
                        # Direct image data format
                        elif part.get("type") == "image" and part.get("data"):
                            logger.info("✅ Image generated via Nanobana (image data format)")
                            return base64.b64decode(part.get("data"))

            # Format 2: content is a base64 data URL string
            if isinstance(content, str) and content.startswith("data:image"):
                base64_data = content.split(",", 1)[1]
                logger.info("✅ Image generated via Nanobana (data URL format)")
                return base64.b64decode(base64_data)

            # Format 3: Check for 'images' field in message (OpenRouter image models use this)
            images = message.get("images", [])
            if images and len(images) > 0:
                image_item = images[0]
                
                # Handle dict format: {"type": "image_url", "image_url": {"url": "data:image/..."}}
                if isinstance(image_item, dict):
                    if image_item.get("type") == "image_url":
                        url = image_item.get("image_url", {}).get("url", "")
                        if url.startswith("data:image"):
                            base64_data = url.split(",", 1)[1]
                            logger.info("✅ Image generated via Nanobana (images field - image_url)")
                            return base64.b64decode(base64_data)
                    # Handle other dict formats
                    for key in ["url", "data", "b64_json", "base64"]:
                        if key in image_item:
                            val = image_item[key]
                            if isinstance(val, str):
                                if val.startswith("data:image"):
                                    base64_data = val.split(",", 1)[1]
                                else:
                                    base64_data = val
                                logger.info(f"✅ Image generated via Nanobana (images field - {key})")
                                return base64.b64decode(base64_data)
                
                # Handle string format (raw base64 or data URL)
                elif isinstance(image_item, str):
                    if image_item.startswith("data:image"):
                        base64_data = image_item.split(",", 1)[1]
                    else:
                        base64_data = image_item
                    logger.info("✅ Image generated via Nanobana (images field - string)")
                    return base64.b64decode(base64_data)

            # If content is text, log it for debugging
            content_preview = str(content)[:300] if content else 'empty'
            logger.warning(f"Nanobana returned unexpected format: {content_preview}")
            logger.debug(f"Full response: {json.dumps(data, indent=2)[:1000]}")
            return None

        except httpx.HTTPStatusError as e:
            error_text = e.response.text[:500] if e.response.text else "No response body"
            logger.error(f"Nanobana HTTP error: {e.response.status_code} - {error_text}")
            return None
        except Exception as e:
            logger.error(f"Nanobana image generation failed: {e}")
            return None


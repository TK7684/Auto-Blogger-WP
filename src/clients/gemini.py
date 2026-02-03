import logging
import os
from pathlib import Path
from typing import Optional, Any, Dict
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.genai.errors import ClientError

logger = logging.getLogger(__name__)

class GeminiClient:
    """Centralized client for Gemini API interactions.

    Supports regional endpoint rotation for quota distribution.
    """

    # Available regions for Vertex AI (in order of preference)
    REGIONS = ["us-central1", "us-east1", "us-west1", "europe-west1", "asia-southeast1"]

    def __init__(self, api_key: Optional[str] = None, service_account_file: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.service_account_file = service_account_file or os.environ.get("GEMINI_SERVICE_ACCOUNT_KEY_FILE")
        self._using_vertexai = False  # Track if using Vertex AI
        self._current_region_index = 0  # For region rotation
        self._credentials = None  # Store credentials for region switching
        self._project_id = None  # Store project ID for region switching
        self.client = self._initialize_client()

    def _initialize_client(self) -> Optional[genai.Client]:
        """Initialize the GenAI client with preferred authentication."""
        try:
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
                    # Start with the first region
                    return self._create_client_for_region(self.REGIONS[0])
                else:
                    logger.warning(f"Service account file not found: {resolved_path}")

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
        """Rotate to the next region and return the region name.

        Returns:
            The new region name.
        """
        self._current_region_index = (self._current_region_index + 1) % len(self.REGIONS)
        new_region = self.REGIONS[self._current_region_index]
        logger.info(f"Rotating to region: {new_region}")
        # Recreate client with new region
        self.client = self._create_client_for_region(new_region)
        return new_region

    def get_current_region(self) -> str:
        """Get the current region being used."""
        return self.REGIONS[self._current_region_index]

    def is_using_vertexai(self) -> bool:
        """Check if the client is using Vertex AI authentication."""
        return self._using_vertexai

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=20),
        retry=retry_if_exception_type(ClientError),
        reraise=True
    )
    def generate_content(self, model: str, contents: Any, config: Optional[types.GenerateContentConfig] = None) -> Any:
        """Generate content with retry logic for API errors."""
        if not self.client:
            raise RuntimeError("Gemini client not initialized")

        logger.info(f"Calling Gemini API (Model: {model})")
        return self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )

    def generate_structured_output(self, model: str, prompt: str, schema: Dict, tools: Optional[list] = None) -> Optional[Any]:
        """Generate content expected to match a specific JSON schema."""
        config = types.GenerateContentConfig(
            tools=tools,
            response_mime_type="application/json",
            response_json_schema=schema,
            temperature=1.0
        )
        try:
            response = self.generate_content(model, prompt, config)
            return response
        except Exception as e:
            logger.error(f"Structured output generation failed: {e}")
            return None

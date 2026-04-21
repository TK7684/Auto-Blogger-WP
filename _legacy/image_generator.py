"""
Image Generator Module.
Optimized for Gemini and Hugging Face.
"""

import os
import logging
import requests
import base64
import time
from google import genai
from google.genai import types
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Create a session for connection pooling
_session = requests.Session()
_session.verify = True

class ImageGenerator:
    """Generate featured images using various AI services."""

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.hf_token = os.environ.get("HUGGINGFACE_API_KEY")
        self.client = None
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")

        self.max_size = int(os.environ.get("MAX_IMAGE_SIZE", "1920"))

    def generate_image(self, prompt: str, mode: str = "daily") -> Optional[bytes]:
        """Priority: Gemini -> Hugging Face -> DALL-E."""

        # 1. Try Gemini (if available)
        if self.client:
            try:
                logger.info("📸 Attempting Gemini image generation...")
                response = self.client.models.generate_content(
                    model="imagen-3.0-generate-001",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        aspect_ratio="16:9"
                    )
                )
                image_parts = [p for p in response.parts if p.inline_data]
                if image_parts:
                    logger.info("✅ Image generated via Gemini")
                    return image_parts[0].inline_data.data
            except Exception as e:
                logger.warning(f"Gemini image generation failed: {e}")

        # 2. Try Hugging Face (if token available)
        if self.hf_token:
            try:
                logger.info("📸 Attempting Hugging Face generation...")
                return self.generate_image_huggingface(prompt)
            except Exception as e:
                logger.warning(f"Hugging Face failed: {e}")

        # 3. Fallback to DALL-E (if API key available)
        return self.generate_image_dalle(prompt)

    def generate_image_huggingface(self, prompt: str) -> Optional[bytes]:
        """Generate image using Hugging Face Inference API."""
        api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {"Authorization": f"Bearer {self.hf_token}"}

        try:
            response = _session.post(api_url, headers=headers, json={"inputs": prompt}, timeout=60)
            if response.status_code == 200:
                logger.info("✅ Image generated via Hugging Face")
                return response.content
            else:
                logger.error(f"HF Error {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"HF Exception: {e}")
        return None

    def generate_image_dalle(self, prompt: str) -> Optional[bytes]:
        """Alternative: DALL-E API."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key: return None
        try:
            import openai
            openai.api_key = api_key
            logger.info("Generating with DALL-E...")
            response = openai.Image.create(prompt=prompt, n=1, size="1024x1024", response_format="b64_json")
            return base64.b64decode(response.data[0].b64_json)
        except Exception as e:
            logger.error(f"DALL-E failed: {e}")
        return None

    def save_image(self, image_data: bytes, filename: str) -> Optional[str]:
        """Save image data to a file in the generated_images directory."""
        output_dir = Path("generated_images")
        output_dir.mkdir(exist_ok=True)  # Create directory if it doesn't exist

        filepath = output_dir / filename
        try:
            with open(filepath, 'wb') as f:
                f.write(image_data)
            logger.info(f"✅ Image saved to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
        return None


class WordPressMediaUploader:
    """Upload images to WordPress media library."""

    def __init__(self, wp_url: str, wp_user: str, wp_app_password: str):
        self.wp_url = wp_url.rstrip('/')
        self.credentials = f"{wp_user}:{wp_app_password}"

    def upload_media(self, file_path: str, alt_text: str = "", title: str = "") -> Optional[int]:
        """Upload an image to WordPress and return the attachment ID."""
        import base64

        url = f"{self.wp_url}/wp-json/wp/v2/media"
        token = base64.b64encode(self.credentials.encode()).decode('utf-8')
        headers = {
            "Authorization": f"Basic {token}",
            "Content-Type": "image/jpeg"
        }

        try:
            with open(file_path, 'rb') as f:
                image_data = f.read()

            response = _session.post(
                url,
                headers=headers,
                data=image_data,
                params={"alt_text": alt_text, "title": title},
                timeout=30
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

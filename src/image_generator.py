"""
Image Generator Module.
Optimized for standardized Gemini client and Hugging Face.
"""

import os
import logging
import requests
import base64
from typing import Optional, Tuple
from pathlib import Path
from .clients.gemini import GeminiClient
from .clients.wordpress import WordPressClient
from google.genai import types

logger = logging.getLogger(__name__)

class ImageGenerator:
    """Generate featured images using various AI services."""

    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        self.gemini_client = gemini_client
        self.hf_token = os.environ.get("HUGGINGFACE_API_KEY")

    def generate_image(self, prompt: str, mode: str = "daily") -> Optional[bytes]:
        """Priority: Gemini 3 Pro Image -> Hugging Face -> DALL-E."""
        
        # 1. Try Gemini 3 Pro Image
        if self.gemini_client and self.gemini_client.client:
            try:
                logger.info("📸 Attempting Gemini 3 Pro Image generation...")
                response = self.gemini_client.client.models.generate_content(
                    model="gemini-3-pro-image-preview",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=[{"google_search": {}}],
                        image_config=types.ImageConfig(
                            aspect_ratio="16:9",
                            image_size="LARGE"
                        )
                    )
                )
                image_parts = [p for p in response.parts if p.inline_data]
                if image_parts:
                    logger.info("✅ Image generated via Gemini 3")
                    return image_parts[0].inline_data.data
            except Exception as e:
                logger.warning(f"Gemini 3 Image failed: {e}")

        # 2. Try Hugging Face
        if self.hf_token:
            try:
                logger.info("📸 Attempting Hugging Face generation...")
                return self.generate_image_huggingface(prompt)
            except Exception as e:
                logger.warning(f"Hugging Face failed: {e}")

        # 3. Fallback to DALL-E (legacy)
        return self.generate_image_dalle(prompt)

    def generate_image_huggingface(self, prompt: str) -> Optional[bytes]:
        api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        try:
            response = requests.post(api_url, headers=headers, json={"inputs": prompt}, timeout=60)
            if response.status_code == 200:
                logger.info("✅ Image generated via Hugging Face")
                return response.content
        except Exception as e:
            logger.error(f"HF Exception: {e}")
        return None

    def generate_image_dalle(self, prompt: str) -> Optional[bytes]:
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

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
from google.genai import types

logger = logging.getLogger(__name__)

class ImageGenerator:
    """Generate featured images using various AI services."""

    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        self.gemini_client = gemini_client
        self.hf_token = os.environ.get("HUGGINGFACE_API_KEY")
        self.session = requests.Session()

    def generate_image(self, prompt: str, mode: str = "daily") -> Optional[bytes]:
        """Priority: Gemini 3 Pro Image -> Hugging Face -> DALL-E."""
        
        # 1. Try Gemini 3 Pro Image (and fallbacks)
        if self.gemini_client and self.gemini_client.client:
            models_to_try = ["imagen-3.0-generate-001"]
            
            for model_name in models_to_try:
                try:
                    logger.info(f"📸 Attempting Gemini Image generation with {model_name}...")
                    response = self.gemini_client.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            image_config=types.ImageConfig(
                                aspect_ratio="16:9"
                            )
                        )
                    )
                    image_parts = [p for p in response.parts if p.inline_data]
                    if image_parts:
                        logger.info(f"✅ Image generated via {model_name}")
                        return image_parts[0].inline_data.data
                except Exception as e:
                    logger.warning(f"Gemini Image {model_name} failed: {e}")

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
        api_url = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        try:
            response = self.session.post(api_url, headers=headers, json={"inputs": prompt}, timeout=60)
            if response.status_code == 200:
                logger.info("✅ Image generated via Hugging Face")
                return response.content
            else:
                logger.warning(f"HF Failed: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"HF Exception: {e}")
        return None

    def generate_image_dalle(self, prompt: str) -> Optional[bytes]:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key: return None
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            logger.info("Generating with DALL-E 3...")
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1024x1024",
                response_format="b64_json"
            )
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

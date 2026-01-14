"""
Image Generator Module.
Optimized for Gemini 3 (Imagen 3) and Hugging Face.
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

class ImageGenerator:
    """Generate featured images using various AI services."""

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.hf_token = os.environ.get("HUGGINGFACE_API_KEY")
        self.client = None
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)

        self.max_size = int(os.environ.get("MAX_IMAGE_SIZE", "1920"))

    def generate_image(self, prompt: str, mode: str = "daily") -> Optional[bytes]:
        """Priority: Gemini 3 Pro Image -> Hugging Face -> DALL-E."""
        
        # 1. Try Gemini 3 Pro Image (Nano Banana Pro)
        # Features: Grounded by Google Search, 4K resolution
        if self.client:
            try:
                logger.info("📸 Attempting Gemini 3 Pro Image generation...")
                response = self.client.models.generate_content(
                    model="gemini-3-pro-image-preview",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=[{"google_search": {}}],
                        image_config=types.ImageConfig(
                            aspect_ratio="16:9"
                        )
                    )
                )
                image_parts = [p for p in response.parts if p.inline_data]
                if image_parts:
                    logger.info("✅ Image generated via Gemini 3")
                    return image_parts[0].inline_data.data
            except Exception as e:
                logger.warning(f"Gemini 3 Image failed: {e}")

        # 2. Try Hugging Face (requested by user)
        if self.hf_token:
            try:
                logger.info("📸 Attempting Hugging Face generation...")
                return self.generate_image_huggingface(prompt)
            except Exception as e:
                logger.warning(f"Hugging Face failed: {e}")

        # 3. Fallback to DALL-E (legacy)
        return self.generate_image_dalle(prompt)

    def generate_image_huggingface(self, prompt: str) -> Optional[bytes]:
        """Generate image using Hugging Face Inference API."""
        # Using Stable Diffusion XL as a high-quality model
        api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        
        try:
            response = requests.post(api_url, headers=headers, json={"inputs": prompt}, timeout=60)
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
        """Save image data to file."""
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
    """Upload images to WordPress."""
    def __init__(self, wp_url: str, wp_user: str, wp_app_password: str):
        self.wp_url = wp_url.rstrip('/')
        self.wp_user = wp_user
        self.wp_app_password = wp_app_password

    def upload_media(self, image_path: str, alt_text: str, title: str) -> Optional[int]:
        try:
            with open(image_path, 'rb') as f: image_data = f.read()
            credentials = f"{self.wp_user}:{self.wp_app_password}"
            token = base64.b64encode(credentials.encode()).decode('utf-8')
            headers = {
                "Authorization": f"Basic {token}",
                "Content-Type": "image/jpeg",
                "Content-Disposition": f'attachment; filename="{os.path.basename(image_path)}"'
            }
            url = f"{self.wp_url}/wp-json/wp/v2/media"
            response = requests.post(url, headers=headers, data=image_data, timeout=60)
            if response.status_code == 201:
                mid = response.json().get('id')
                # Update meta
                requests.post(f"{url}/{mid}", headers=headers, json={"alt_text": alt_text, "title": title})
                return mid
        except Exception as e:
            logger.error(f"Upload error: {e}")
        return None

"""
Test script for Nanobana image generation via OpenRouter.

Run with: python tests/test_nanobana.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.clients.gemini import GeminiClient
from src.image_generator import ImageGenerator


def test_nanobana_generation():
    """Test Nanobana image generation via OpenRouter."""
    print("=" * 60)
    print("TEST: Nanobana Image Generation")
    print("=" * 60)

    # Initialize clients
    gemini_client = GeminiClient()

    if not gemini_client.is_using_openrouter():
        print("❌ OpenRouter not configured. Set OPENROUTER_API_KEY in .env")
        return False

    print("✅ OpenRouter mode detected")

    # Initialize image generator
    image_gen = ImageGenerator(gemini_client=gemini_client)

    # Test with a simple prompt
    prompt = "A modern blog header image about digital marketing and SEO optimization"

    print(f"\n📤 Testing with prompt: {prompt[:50]}...")

    try:
        # Use gemini-3-pro-image which is confirmed working
        image_bytes = image_gen.generate_image_nanobana(prompt, model="gemini-3-pro-image")

        if image_bytes:
            # Save test image
            test_path = "generated_images/test_nanobana.png"
            os.makedirs("generated_images", exist_ok=True)
            with open(test_path, "wb") as f:
                f.write(image_bytes)
            print(f"✅ Image generated successfully!")
            print(f"   Size: {len(image_bytes)} bytes")
            print(f"   Saved to: {test_path}")
            return True
        else:
            print("❌ No image bytes returned")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_content_aware_generation():
    """Test content-aware image generation."""
    print("\n" + "=" * 60)
    print("TEST: Content-Aware Image Generation")
    print("=" * 60)

    gemini_client = GeminiClient()

    if not gemini_client.is_using_openrouter():
        print("⏭️ Skipping: OpenRouter not configured")
        return None

    image_gen = ImageGenerator(gemini_client=gemini_client)

    # Sample blog content
    content = """
    Understanding the Power of Internal Linking for SEO

    Internal linking is one of the most underutilized SEO strategies.
    By connecting related content on your website, you help search engines
    understand your site structure and distribute page authority effectively.

    Key benefits include improved crawlability, better user engagement,
    and enhanced topical relevance signals.
    """

    prompt = "Blog featured image"

    print(f"📤 Generating with content context...")

    try:
        image_bytes = image_gen.generate_image(
            prompt=prompt,
            content_context=content
        )

        if image_bytes:
            test_path = "generated_images/test_content_aware.png"
            with open(test_path, "wb") as f:
                f.write(image_bytes)
            print(f"✅ Content-aware image generated!")
            print(f"   Size: {len(image_bytes)} bytes")
            print(f"   Saved to: {test_path}")
            return True
        else:
            print("❌ No image generated")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n🎨 Nanobana Image Generation Tests\n")

    results = {
        "Nanobana Generation": test_nanobana_generation(),
        "Content-Aware": test_content_aware_generation(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    for name, result in results.items():
        status = "✅ PASSED" if result is True else ("❌ FAILED" if result is False else "⏭️ SKIPPED")
        print(f"{status}: {name}")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

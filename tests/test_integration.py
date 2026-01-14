"""
Integration tests for Auto-Blogging WordPress application.
Updated for google-genai and Gemini 3.
"""

import os
import sys
import io
import unittest
import base64
import requests
import feedparser
from google import genai
from dotenv import load_dotenv

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Load environment variables
load_dotenv()


class TestGeminiAPIIntegration(unittest.TestCase):
    """Test real Gemini API connectivity and content generation."""

    @classmethod
    def setUpClass(cls):
        """Set up client for all tests."""
        cls.api_key = os.environ.get("GEMINI_API_KEY")
        if cls.api_key:
            cls.client = genai.Client(api_key=cls.api_key)

    def test_01_gemini_api_key_exists(self):
        """Test that GEMINI_API_KEY is configured."""
        self.assertIsNotNone(
            self.api_key,
            "GEMINI_API_KEY not found in environment. Please set it in .env file."
        )
        print(f"✅ Gemini API Key found: {self.api_key[:10]}...{self.api_key[-4:]}")

    def test_02_gemini_flash_daily_content_generation(self):
        """Test actual content generation with Gemini 3 Flash model."""
        if not self.api_key:
            self.skipTest("GEMINI_API_KEY not configured")

        print("\n🧠 Testing Gemini 3 Flash (gemini-3-flash-preview) for daily content...")
        try:
            response = self.client.models.generate_content(
                model="gemini-3-flash-preview",
                contents="Write a short 100-word test article about artificial intelligence."
            )

            self.assertIsNotNone(response.text, "Response text should not be None")
            self.assertGreater(len(response.text), 50, "Generated content should be meaningful")
            print(f"✅ Flash generated {len(response.text)} characters")
        except Exception as e:
            self.fail(f"Gemini 3 Flash API call failed: {e}")

    def test_03_gemini_pro_weekly_content_generation(self):
        """Test actual content generation with Gemini 3 Pro model."""
        if not self.api_key:
            self.skipTest("GEMINI_API_KEY not configured")

        print("\n🧠 Testing Gemini 3 Pro (gemini-3-pro-preview) for weekly content...")
        try:
            response = self.client.models.generate_content(
                model="gemini-3-pro-preview",
                contents="Write a short 150-word deep dive about quantum computing."
            )

            self.assertIsNotNone(response.text, "Response text should not be None")
            self.assertGreater(len(response.text), 100, "Generated content should be meaningful")
            print(f"✅ Pro generated {len(response.text)} characters")
        except Exception as e:
            self.fail(f"Gemini 3 Pro API call failed: {e}")


class TestWordPressAPIIntegration(unittest.TestCase):
    """Test real WordPress API connectivity."""

    @classmethod
    def setUpClass(cls):
        """Set up WordPress credentials for all tests."""
        cls.wp_url = os.environ.get("WP_URL")
        cls.wp_user = os.environ.get("WP_USER")
        cls.wp_app_password = os.environ.get("WP_APP_PASSWORD")

    def test_01_wordpress_config_exists(self):
        """Test that WordPress credentials are configured."""
        missing = []
        if not self.wp_url:
            missing.append("WP_URL")
        if not self.wp_user:
            missing.append("WP_USER")
        if not self.wp_app_password:
            missing.append("WP_APP_PASSWORD")

        if missing:
            self.fail(f"WordPress config incomplete. Missing: {', '.join(missing)}")

        print(f"✅ WordPress config found")

    def test_02_wordpress_api_connection(self):
        """Test WordPress REST API connectivity."""
        if not all([self.wp_url, self.wp_user, self.wp_app_password]):
            self.skipTest("WordPress credentials not configured")

        print("\n🔌 Testing WordPress REST API connection...")
        try:
            url = f"{self.wp_url.rstrip('/')}/wp-json/wp/v2/posts"
            credentials = f"{self.wp_user}:{self.wp_app_password}"
            token = base64.b64encode(credentials.encode()).decode('utf-8')
            headers = {
                "Authorization": f"Basic {token}",
                "Content-Type": "application/json"
            }

            response = requests.get(url, headers=headers, timeout=30)
            self.assertEqual(response.status_code, 200, f"Expected 200, got {response.status_code}")
            print(f"✅ WordPress API connection successful!")

        except Exception as e:
            self.fail(f"WordPress API test failed: {e}")


class TestGoogleTrendsIntegration(unittest.TestCase):
    """Test Google Trends RSS feed fetching."""

    def test_01_google_trends_rss_fetch(self):
        """Test fetching Google Trends RSS feed."""
        geo = os.environ.get("TRENDS_GEO", "US")
        rss_url = f"https://trends.google.com/trends/trendingsearches/daily/rss?geo={geo}"

        print(f"\n🔍 Testing Google Trends RSS feed for geo={geo}...")

        try:
            feed = feedparser.parse(rss_url)
            self.assertIsNotNone(feed, "Feed should not be None")
            if not feed.entries:
                self.skipTest("Google Trends RSS currently empty or unavailable")
            print(f"✅ RSS feed fetched successfully!")
        except unittest.SkipTest:
            raise
        except Exception as e:
            self.fail(f"Failed to fetch Google Trends RSS: {e}")


def run_integration_tests():
    """Run integration tests with detailed output."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestGeminiAPIIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestWordPressAPIIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestGoogleTrendsIntegration))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == '__main__':
    run_integration_tests()

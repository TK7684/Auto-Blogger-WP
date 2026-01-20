"""
Yoast SEO Integration Module.

Handles integration with Yoast SEO plugin for WordPress:
- Updates Yoast meta fields
- Calculates readability and SEO scores
- Optimizes content for Yoast's analysis
"""

import logging
import requests
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class YoastSEOIntegrator:
    """Integrate with Yoast SEO plugin meta fields."""

    def __init__(self, wp_url: str, wp_user: str, wp_app_password: str):
        self.wp_url = wp_url.rstrip('/')
        self.wp_user = wp_user
        self.wp_app_password = wp_app_password
        self.api_base = f"{self.wp_url}/wp-json"

    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for WordPress REST API."""
        import base64
        credentials = f"{self.wp_user}:{self.wp_app_password}"
        token = base64.b64encode(credentials.encode()).decode('utf-8')
        return {
            "Authorization": f"Basic {token}",
            "Content-Type": "application/json"
        }

    def update_yoast_meta_fields(self, post_id: int, seo_data: Dict) -> bool:
        """
        Update Yoast SEO meta fields for a post via standard WP Meta API.
        This avoids 404 errors from the specific Yoast REST API if it's not fully enabled.
        """
        meta_fields = {
            "_yoast_wpseo_focuskw": seo_data.get('focus_keyword', ''),
            "_yoast_wpseo_title": seo_data.get('seo_title', ''),
            "_yoast_wpseo_metadesc": seo_data.get('meta_description', ''),
            "_yoast_wpseo_canonical": seo_data.get('canonical_url', ''),
            "_yoast_wpseo_opengraph-title": seo_data.get('og_title', ''),
            "_yoast_wpseo_opengraph-description": seo_data.get('og_description', ''),
            "_yoast_wpseo_opengraph-image": seo_data.get('og_image', ''),
            "_yoast_wpseo_linkdex": str(seo_data.get('seo_score', 0)),
            "_yoast_wpseo_content_score": str(seo_data.get('readability_score', 0)),
        }

        # Filter out empty strings if necessary, but usually better to send what we have
        meta_payload = {k: v for k, v in meta_fields.items() if v}

        try:
            url = f"{self.wp_url}/wp-json/wp/v2/posts/{post_id}"
            headers = self.get_auth_headers()
            
            # Use the "meta" key in the post update payload
            response = requests.post(url, headers=headers, json={"meta": meta_payload}, timeout=30)
            
            if response.status_code == 200:
                logger.info(f"✅ Yoast SEO meta fields updated for post {post_id}")
                return True
            else:
                logger.warning(f"❌ Failed to update Yoast meta for post {post_id}: {response.status_code} - {response.text}")
                # Fallback to individual updates if needed (though unlikely to help if 404)
                return False

        except Exception as e:
            logger.error(f"Error updating Yoast meta fields: {e}")
            return False

    def update_meta_fallback(self, post_id: int, seo_data: Dict) -> bool:
        """Fallback: Update meta fields via standard WP API if Yoast API fails."""
        meta_fields = {
            "_yoast_wpseo_focuskw": seo_data.get('focus_keyword', ''),
            "_yoast_wpseo_title": seo_data.get('seo_title', ''),
            "_yoast_wpseo_metadesc": seo_data.get('meta_description', ''),
        }
        
        url = f"{self.api_base}/wp/v2/posts/{post_id}"
        headers = self.get_auth_headers()
        
        try:
            response = requests.post(url, headers=headers, json={"meta": meta_fields}, timeout=20)
            if response.status_code == 200:
                logger.info(f"Fallback: Updated Yoast meta for post {post_id}")
                return True
        except Exception as e:
            logger.error(f"Fallback meta update failed: {e}")
        return False

    def calculate_seo_score(self, content: str, focus_keyword: str,
                           title: str, meta_description: str) -> int:
        """
        Calculate an approximate SEO score based on Yoast's criteria.

        Args:
            content: Article content (HTML)
            focus_keyword: Primary focus keyword
            title: SEO title
            meta_description: Meta description

        Returns:
            SEO score (0-100)
        """
        score = 0
        content_lower = content.lower()
        keyword_lower = focus_keyword.lower()

        # Keyword in title (10 points)
        if keyword_lower in title.lower():
            score += 10

        # Keyword in first paragraph (10 points)
        first_paragraph = content_lower.split('\n')[0] if content_lower else ''
        if keyword_lower in first_paragraph:
            score += 10

        # Keyword density (2-3% ideal) (20 points)
        word_count = len(content_lower.split())
        keyword_count = content_lower.count(keyword_lower)
        if word_count > 0:
            density = (keyword_count / word_count) * 100
            if 0.5 <= density <= 3:
                score += 20
            elif density > 0:
                score += 10

        # Content length (20 points)
        if word_count >= 300:
            score += 20
        elif word_count >= 200:
            score += 10

        # Meta description length and keyword (15 points)
        if 120 <= len(meta_description) <= 160:
            score += 10
        if keyword_lower in meta_description.lower():
            score += 5

        # Title length (15 points)
        if 30 <= len(title) <= 60:
            score += 15
        elif len(title) < 60:
            score += 10

        # Subheadings (10 points)
        if '<h2>' in content or '<h3>' in content:
            score += 10

        # Links (5 points)
        if '<a href=' in content:
            score += 5

        return min(score, 100)

    def calculate_readability_score(self, content: str) -> int:
        """
        Calculate readability score based on Yoast's criteria.

        Args:
            content: Article content (HTML)

        Returns:
            Readability score (0-100)
        """
        # Remove HTML tags for analysis
        import re
        text_content = re.sub(r'<[^>]+>', ' ', content)
        text_content = ' '.join(text_content.split())

        if not text_content:
            return 0

        score = 0

        # Sentence length (30 points)
        sentences = text_content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

        if avg_sentence_length <= 20:
            score += 30
        elif avg_sentence_length <= 25:
            score += 20
        else:
            score += 10

        # Paragraph length (30 points)
        paragraphs = text_content.split('\n\n')
        short_paragraphs = sum(1 for p in paragraphs if len(p.split()) <= 150)
        if len(paragraphs) > 0:
            paragraph_ratio = short_paragraphs / len(paragraphs)
            if paragraph_ratio >= 0.7:
                score += 30
            elif paragraph_ratio >= 0.5:
                score += 20

        # Transition words (20 points)
        transitions = {'however', 'therefore', 'moreover', 'furthermore', 'additionally',
                      'consequently', 'meanwhile', 'nevertheless', 'thus', 'hence'}
        transition_count = sum(1 for word in text_content.lower().split() if word in transitions)
        if len(text_content.split()) > 0:
            transition_ratio = transition_count / len(text_content.split())
            if transition_ratio >= 0.1:
                score += 20
            elif transition_ratio >= 0.05:
                score += 10

        # Passive voice detection (20 points) - simplified
        passive_indicators = {'was', 'were', 'been', 'being', 'is', 'are', 'am'}
        words = text_content.lower().split()
        passive_count = sum(1 for i, word in enumerate(words)
                           if word in passive_indicators
                           and i + 1 < len(words)
                           and words[i + 1].endswith('ed'))

        if len(words) > 0:
            passive_ratio = passive_count / len(words)
            if passive_ratio <= 0.05:
                score += 20
            elif passive_ratio <= 0.1:
                score += 10

        return min(score, 100)


class ContentAnalyzer:
    """Analyze content for SEO and readability optimization."""

    def __init__(self):
        pass

    def analyze_keyword_density(self, content: str, keyword: str) -> Dict:
        """Analyze keyword density in content."""
        words = content.lower().split()
        keyword_lower = keyword.lower()
        keyword_count = content.lower().count(keyword_lower)

        density = (keyword_count / len(words) * 100) if words else 0

        return {
            'keyword_count': keyword_count,
            'total_words': len(words),
            'density_percentage': round(density, 2),
            'status': 'good' if 0.5 <= density <= 3 else 'low' if density < 0.5 else 'high'
        }

    def suggest_subheadings(self, content: str) -> List[str]:
        """Suggest where to add subheadings for better structure."""
        # Find long paragraphs without subheadings
        paragraphs = content.split('\n\n')
        suggestions = []

        for i, para in enumerate(paragraphs):
            word_count = len(para.split())
            if word_count > 150 and '<h' not in para:
                suggestions.append(f"Paragraph {i + 1} ({word_count} words): Consider adding a subheading")

        return suggestions

    def extract_key_phrases(self, content: str, min_freq: int = 2) -> List[str]:
        """Extract frequently occurring key phrases."""
        import re
        from collections import Counter

        # Remove HTML and get words
        text = re.sub(r'<[^>]+>', ' ', content)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

        # Get common phrases (2-3 word combinations)
        phrases = []
        for i in range(len(words) - 1):
            phrases.append(f"{words[i]} {words[i+1]}")

        phrase_freq = Counter(phrases)
        key_phrases = [p for p, freq in phrase_freq.items() if freq >= min_freq]

        return key_phrases[:10]


if __name__ == "__main__":
    # Test Yoast integration
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    wp_url = os.environ.get("WP_URL")
    wp_user = os.environ.get("WP_USER")
    wp_app_password = os.environ.get("WP_APP_PASSWORD")

    if all([wp_url, wp_user, wp_app_password]):
        integrator = YoastSEOIntegrator(wp_url, wp_user, wp_app_password)

        # Test scoring
        test_content = """
        <h2>Test Heading</h2>
        <p>This is a test paragraph about artificial intelligence and machine learning.
        AI is transforming the world.</p>
        """

        seo_score = integrator.calculate_seo_score(
            test_content,
            "artificial intelligence",
            "Complete Guide to Artificial Intelligence",
            "Learn everything about artificial intelligence and how it's changing the world."
        )

        print(f"SEO Score: {seo_score}/100")
    else:
        print("Missing WordPress credentials")

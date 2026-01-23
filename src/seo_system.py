"""
Comprehensive SEO System for Auto-Blogging WordPress.

This module provides:
- SEO-optimized content prompts
- Schema.org JSON-LD markup generation
- SERP analysis and competitor insights
- Keyword extraction and optimization
"""

import os
import json
import logging
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class SEOPromptBuilder:
    """Builds SEO-optimized prompts for content generation."""

    def __init__(self):
        self.site_name = os.environ.get("SITE_NAME", "PedPro")
        self.site_url = os.environ.get("SITE_URL", "https://pedpro.online")
        self.author_name = os.environ.get("DEFAULT_AUTHOR_NAME", "AI Author")
        self.guidelines_path = "brand_guidelines.json"
        self._load_guidelines()

    def _load_guidelines(self):
        """Load brand guidelines from JSON file."""
        self.guidelines = []
        if os.path.exists(self.guidelines_path):
            try:
                with open(self.guidelines_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.guidelines = data.get("guidelines", [])
            except Exception as e:
                logger.error(f"Error loading guidelines: {e}")

    def get_applicable_guidelines(self, topic: str, context: str) -> List[str]:
        """Find guidelines that match keywords in the topic or context."""
        text = (topic + " " + context).upper()
        active_facts = []
        for g in self.guidelines:
            if any(kw.upper() in text for kw in g.get("keywords", [])):
                active_facts.extend(g.get("strict_facts", []))
        return list(set(active_facts))

    def extract_focus_keyword(self, topic: str) -> str:
        """Extract the primary focus keyword from the topic."""
        # Remove common stopwords and get the main phrase
        stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'must', 'what', 'when', 'where', 'why', 'how',
                     'and', 'or', 'but', 'if', 'then', 'else', 'so', 'because', 'although'}

        words = topic.lower().split()
        keywords = [w for w in words if w not in stopwords and len(w) > 2]

        # Return the most significant 2-3 word phrase
        if len(keywords) >= 2:
            return ' '.join(keywords[:2])
        elif keywords:
            return keywords[0]
        return topic

    def build_daily_prompt(self, topic: str, context: str,
                          competitor_insights: Optional[str] = None,
                          language: str = "English") -> str:
        """
        Build an SEO-optimized prompt for daily content.

        Args:
            topic: The trending topic
            context: Additional context about the topic
            competitor_insights: Optional SERP competitor analysis

        Returns:
            SEO-optimized prompt string
        """
        focus_keyword = self.extract_focus_keyword(topic)
        brand_facts = self.get_applicable_guidelines(topic, context)
        brand_instruction = ""
        if brand_facts:
            brand_instruction = "\nSTRICT BRAND GUIDELINES (Must follow for factual accuracy):\n- " + "\n- ".join(brand_facts) + "\n"

        prompt = f"""
You are an expert SEO content writer and journalist. Write a comprehensive blog post about:

TOPIC: {topic}
CONTEXT: {context}
FOCUS KEYWORD: {focus_keyword}
LANGUAGE: {language}
{brand_instruction}

IMPORTANT: Write the entire content (including titles, headings, and body) in {language}. Keep the JSON keys (like SEO_TITLE, IMAGE_PROMPT) in English, but their values must be in {language}.

Requirements:

1. SEO META GENERATION (Start your response with these exact lines):
```
SEO_TITLE: [URGENT: Create a massive click-bait style title but professional. MUST grab attention in 3 seconds. Max 60 chars. Use power words. Example: "Why X is Changing Everything", "Stop Doing Y Immediately". Focus Keyword: "{focus_keyword}"]
META_DESCRIPTION: [155-160 chars. deeply engaging, teases the solution, includes "{focus_keyword}"]
FOCUS_KEYWORD: {focus_keyword}
EXCERPT: [2 engaging sentences that force the reader to click 'Read More']
```

2. CONTENT STRUCTURE (HTML format):

H1: {topic} (or the optimized SEO_TITLE)

IMPORTANT: Use the "google_search" tool to verify ALL facts, statistics, and claims. preventing misinformation is your #1 priority.

INTRO (2-3 very short paragraphs):
- Hook the reader IMMEDIATELY (in the first 3 seconds of reading).
- Use a startling fact, a controversial question, or a relatability hook.
- Include "{focus_keyword}" naturally.

BODY (3-4 H2 sections):
- H2s must be benefit-driven (e.g., "How to Save Money" vs "Saving Money").
- Paragraphs must be SHORT (1-2 sentences).
- Use distinct formatting (bolding, italics, lists) to keep the eye moving.
- Include internal link placeholders: [INSERT_INTERNAL_LINK:relevant_topic] where relevant_topics should match existing content on your site.

SEO ELEMENTS:
- Keyword density: 1-2% (natural placement).
- Use LSI keywords.
- **IN-ARTICLE IMAGES**: Include `[IMAGE_PLACEHOLDER_0]` and `[IMAGE_PLACEHOLDER_1]` between sections.
- End with a FAQ section with schema markup.

FAQ SECTION (Schema.org FAQPage format):
```html
<div class="schema-faq" itemscope itemtype="https://schema.org/FAQPage">
  <h2>Frequently Asked Questions About {focus_keyword}</h2>
  <div itemscope itemprop="mainEntity" itemtype="https://schema.org/Question">
    <h3 itemprop="name">[Question about {topic}]?</h3>
    <div itemscope itemprop="acceptedAnswer" itemtype="https://schema.org/Answer">
      <p itemprop="text">[Detailed, helpful answer verified by search]</p>
    </div>
  </div>
  <!-- 2-3 more FAQs -->
</div>
```

LENGTH: 800-1200 words

TONE:
- High energy, engaging, and authoritative.
- "You" focused.
- No fluff. Every sentence must add value.
- **STRICTLY NO INSTRUCTION KEYWORDS**: Never include 'IMAGE_PROMPT:', 'SEO_TITLE:' in the HTML.

OUTPUT:
After the content, provide the image generation prompt:
```
IMAGE_PROMPT: [Hyper-realistic, captivating image for: {topic}. High contrast, shallow depth of field, 8k resolution, trending on ArtStation.]
```
"""

        if competitor_insights:
            prompt += f"""

COMPETITOR INSIGHTS (use to improve your content):
{competitor_insights}

Ensure your content covers these gaps and provides more value than existing articles.
"""

        return prompt

    def build_weekly_prompt(self, topic: str, context: str,
                           competitor_insights: Optional[str] = None,
                           language: str = "English") -> str:
        """
        Build an SEO-optimized prompt for weekly pillar content.

        Args:
            topic: The trending topic
            context: Additional context about the topic
            competitor_insights: Optional SERP competitor analysis

        Returns:
            SEO-optimized prompt string
        """
        focus_keyword = self.extract_focus_keyword(topic)
        brand_facts = self.get_applicable_guidelines(topic, context)
        brand_instruction = ""
        if brand_facts:
            brand_instruction = "\nSTRICT BRAND GUIDELINES (Must follow for factual accuracy):\n- " + "\n- ".join(brand_facts) + "\n"

        prompt = f"""
You are an expert SEO content writer and senior investigative journalist. Write a comprehensive pillar content article about:

TOPIC: {topic}
CONTEXT: {context}
FOCUS KEYWORD: {focus_keyword}
LANGUAGE: {language}
{brand_instruction}

IMPORTANT: Write the entire content (including titles, headings, and body) in {language}. Keep the JSON keys (like SEO_TITLE, IMAGE_PROMPT) in English, but their values must be in {language}.

Requirements:

1. SEO META GENERATION (Start your response with these exact lines):
```
SEO_TITLE: [{focus_keyword}: [Rest of Title]. MUST Start with Focus Keyword. Max 60 chars. High-CTR. Example: "{focus_keyword}: The Ultimate Guide"]
META_DESCRIPTION: [155-160 chars. deeply engaging, teases the solution, includes "{focus_keyword}"]
SLUG: [{focus_keyword}-guide]
FOCUS_KEYWORD: {focus_keyword}
EXCERPT: [3 engaging sentences that force the reader to click 'Read More']
PILLAR_CONTENT: true
```

2. CONTENT STRUCTURE (HTML format):

H1: {topic} (or the optimized SEO_TITLE)

IMPORTANT: Use the "google_search" tool to verify ALL facts, statistics, and claims. Preventing misinformation is your #1 priority.

EXECUTIVE SUMMARY (Styled box):
```html
<div class="wp-block-group has-background" style="background-color:#f0f0f0;padding:25px;border-radius:12px;margin-bottom:30px;border-left:5px solid #0073aa;">
  <h2 style="margin-top:0;">🚀 Executive Summary</h2>
  <p><strong>What you'll learn in 3 minutes:</strong></p>
  <ul>
    <li>Key insight 1 about {focus_keyword}</li>
    <li>Key insight 2 with specific data point</li>
    <li>Actionable takeaway for readers</li>
  </ul>
</div>
```

BODY STRUCTURE (Pillar Content - 1500+ words):

SECTION 1: Introduction & Background
- H1: {topic}
- H2: Understanding {focus_keyword}: The 3-Second Breakdown
- **First Paragraph Usage**: You MUST include "{focus_keyword}" in the very first sentence.
- Hook the reader immediately.
- Cover history, background, and current relevance (Verified Data).

SECTION 2: Deep Dive
- H2: How {focus_keyword} Works: The Technical Details
- H3: Critical Component 1
- H3: Critical Component 2
- Include comparison table:
```html
<table class="wp-block-table is-style-stripes">
  <thead>
    <tr>
      <th>Feature</th>
      <th>Option A</th>
      <th>Option B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Criteria 1</td>
      <td>Detail A</td>
      <td>Detail B</td>
    </tr>
  </tbody>
</table>
```

SECTION 3: Analysis & Implications
- H2: The Real Impact of {focus_keyword} on Industry
- H3: Benefits (The "Why It Matters")
- H3: Challenges (The "What To Watch Out For")
- Include verified case studies.

SECTION 4: Future Outlook
- H2: The Future of {focus_keyword}: Predictions for 2026
- Expert predictions and trends (Verified).
- Timeline graphic placeholder.

SECTION 5: Practical Applications
- H2: How to Leverage {focus_keyword} for Massive Success
- Step-by-step guide.
- Numbered list for clarity.

SECTION 6: FAQs (Verified)
- Ensure the Focus Keyword appears in at least one subheading.
```html
<div class="schema-faq" itemscope itemtype="https://schema.org/FAQPage">
  <h2>Frequently Asked Questions About {focus_keyword}</h2>
  <!-- 5-7 comprehensive FAQs -->
</div>
```

SEO ELEMENTS:
- **Keyword Density**: 1.5-2.5% (use "{focus_keyword}" naturally 15-20 times).
- **Subheadings**: Include "{focus_keyword}" in at least 50% of H2/H3s.
- **Internal Links**: [INSERT_INTERNAL_LINK:relevant_topic] where relevant_topic should match existing content on your site (e.g., 'AI', 'Tech', 'Business').
- **Images**: Include at least 3 placeholders (e.g., `[IMAGE_PLACEHOLDER_0]`).
    - **Alt Text**: Suggest optimized Alt Text for each image containing "{focus_keyword}".

SEMANTIC HTML:
- Use <section> for major divisions
- Use <figure> and <figcaption> for data visualizations
- Use <blockquote> for expert quotes
- Proper heading hierarchy (H1 → H2 → H3)

LENGTH: 1500-2000 words (Critical for SEO)

TONE:
- Authoritative but accessible.
- Data-driven with statistics (Verified).
- High energy.
- **STRICTLY NO INSTRUCTION KEYWORDS**: Never include 'IMAGE_PROMPT:', 'SEO_TITLE:' in the HTML.

OUTPUT:
After the content, provide:
```
IMAGE_PROMPT: [Create a professional, modern 16:9 featured image for: {topic}. Style: premium business/tech publication. High resolution.]
RELATED_TOPICS: [3-5 related topic suggestions for internal linking]
INTERNAL_LINKS: [3-5 internal link opportunities with anchor text]
```
"""

        if competitor_insights:
            prompt += f"""

COMPETITOR INSIGHTS (use to create superior content):
{competitor_insights}

Your content must:
- Be more comprehensive than competitors
- Include data and statistics competitors missed
- Cover angles competitors didn't explore
- Provide unique insights and analysis
"""

        return prompt

    def parse_generated_content(self, response: str) -> Dict[str, str]:
        """
        Parse the AI-generated content to extract SEO metadata and content.

        Args:
            response: The raw response from the AI

        Returns:
            Dictionary with seo_title, meta_description, content, focus_keyword, etc.
        """
        result = {
            'seo_title': None,
            'meta_description': None,
            'focus_keyword': None,
            'excerpt': None,
            'content': response,
            'image_prompt': None,
        }

        # Extract SEO metadata from the response
        lines = response.split('\n')
        content_start = 0

        for i, line in enumerate(lines):
            if line.startswith('SEO_TITLE:'):
                result['seo_title'] = line.split(':', 1)[1].strip()
            elif line.startswith('META_DESCRIPTION:'):
                result['meta_description'] = line.split(':', 1)[1].strip()
            elif line.startswith('FOCUS_KEYWORD:'):
                result['focus_keyword'] = line.split(':', 1)[1].strip()
            elif line.startswith('EXCERPT:'):
                result['excerpt'] = line.split(':', 1)[1].strip()
            elif line.startswith('IMAGE_PROMPT:'):
                result['image_prompt'] = line.split(':', 1)[1].strip()
            elif line.strip().startswith('<h1') or line.strip().startswith('#'):
                content_start = i
                break

        # Extract the main content (without SEO metadata)
        result['content'] = '\n'.join(lines[content_start:]).strip()

        return result


class SchemaMarkupGenerator:
    """Generate Schema.org JSON-LD markup for content."""

    def __init__(self):
        self.site_name = os.environ.get("SITE_NAME", "PedPro")
        self.site_url = os.environ.get("SITE_URL", "https://pedpro.online")
        self.author_name = os.environ.get("DEFAULT_AUTHOR_NAME", "AI Author")
        self.author_url = os.environ.get("DEFAULT_AUTHOR_URL", f"{self.site_url}/author")
        self.locale = os.environ.get("DEFAULT_LOCALE", "en_US")

    def generate_article_schema(self, title: str, description: str,
                                 content: str, url: str,
                                 image_url: Optional[str] = None,
                                 date_published: Optional[str] = None) -> Dict:
        """
        Generate Schema.org Article markup.

        Args:
            title: Article title
            description: Meta description
            content: Article content
            url: Article URL
            image_url: Featured image URL
            date_published: ISO format date

        Returns:
            Schema.org JSON-LD dictionary
        """
        if not date_published:
            date_published = datetime.now(timezone.utc).isoformat()

        schema = {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": title,
            "description": description,
            "url": url,
            "datePublished": date_published,
            "dateModified": date_published,
            "author": {
                "@type": "Person",
                "name": self.author_name,
                "url": self.author_url
            },
            "publisher": {
                "@type": "Organization",
                "name": self.site_name,
                "url": self.site_url,
                "logo": {
                    "@type": "ImageObject",
                    "url": f"{self.site_url}/logo.png"
                }
            },
            "mainEntityOfPage": {
                "@type": "WebPage",
                "@id": url
            }
        }

        if image_url:
            schema["image"] = {
                "@type": "ImageObject",
                "url": image_url,
                "width": "1920",
                "height": "1080"
            }

        # Extract word count
        word_count = len(content.split())
        schema["wordCount"] = word_count

        # Add keywords based on content
        schema["keywords"] = self._extract_keywords(content, title)

        return schema

    def generate_faq_schema(self, faqs: List[Tuple[str, str]]) -> Dict:
        """
        Generate Schema.org FAQPage markup.

        Args:
            faqs: List of (question, answer) tuples

        Returns:
            Schema.org FAQPage JSON-LD dictionary
        """
        faq_entities = []

        for question, answer in faqs:
            faq_entities.append({
                "@type": "Question",
                "name": question,
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": answer
                }
            })

        return {
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "mainEntity": faq_entities
        }

    def generate_breadcrumb_schema(self, breadcrumbs: List[Tuple[str, str]]) -> Dict:
        """
        Generate Schema.org BreadcrumbList markup.

        Args:
            breadcrumbs: List of (name, url) tuples

        Returns:
            Schema.org BreadcrumbList JSON-LD dictionary
        """
        items = []

        for i, (name, url) in enumerate(breadcrumbs, start=1):
            items.append({
                "@type": "ListItem",
                "position": i,
                "name": name,
                "item": url
            })

        return {
            "@context": "https://schema.org",
            "@type": "BreadcrumbList",
            "itemListElement": items
        }

    def extract_faq_from_content(self, content: str) -> List[Tuple[str, str]]:
        """
        Extract FAQ from content with schema markup.

        Args:
            content: HTML content with FAQ section

        Returns:
            List of (question, answer) tuples
        """
        faqs = []

        # Look for FAQ section with schema markup
        import re

        # Pattern to match FAQ schema
        pattern = r'<div[^>]*itemtype="https://schema\.org/Question"[^>]*>.*?<h3[^>]*itemprop="name"[^>]*>(.*?)</h3>.*?<p[^>]*itemprop="text"[^>]*>(.*?)</p>'

        matches = re.findall(pattern, content, re.DOTALL)

        for question, answer in matches:
            # Clean up HTML tags
            question_clean = re.sub(r'<[^>]+>', '', question).strip()
            answer_clean = re.sub(r'<[^>]+>', '', answer).strip()
            faqs.append((question_clean, answer_clean))

        return faqs

    def _extract_keywords(self, content: str, title: str, max_keywords: int = 5) -> List[str]:
        """Extract relevant keywords from content."""
        # Simple keyword extraction (can be enhanced with NLP)
        words = content.lower().split()
        word_freq = {}

        # Filter common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
                     'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that'}

        for word in words:
            word = re.sub(r'[^\w]', '', word)
            if len(word) > 3 and word not in stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Get top keywords
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        # Include title words
        title_words = [w.lower() for w in title.split() if len(w) > 3]
        keywords = list(set(title_words + [k for k, v in sorted_keywords[:max_keywords]]))

        return keywords[:max_keywords]

    def wrap_schema_in_script(self, schema: Dict) -> str:
        """Wrap schema in HTML script tag for injection."""
        return f'<script type="application/ld+json">{json.dumps(schema, indent=2)}</script>'


class KeywordExtractor:
    """Extract and optimize keywords for SEO."""

    def __init__(self):
        pass

    def extract_focus_keyword(self, topic: str) -> str:
        """Extract the primary focus keyword."""
        # Similar to SEOPromptBuilder method
        stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'what', 'when',
                     'where', 'why', 'how', 'and', 'or', 'but', 'for', 'of'}

        words = topic.lower().split()
        keywords = [w for w in words if w not in stopwords and len(w) > 2]

        if len(keywords) >= 2:
            return ' '.join(keywords[:2])
        elif keywords:
            return keywords[0]
        return topic

    def generate_lsi_keywords(self, focus_keyword: str) -> List[str]:
        """Generate LSI (Latent Semantic Indexing) keyword suggestions."""
        # This would ideally use an NLP model or keyword research API
        # For now, return basic variations
        variations = []

        words = focus_keyword.split()
        if len(words) == 2:
            # Reverse order
            variations.append(f"{words[1]} {words[0]}")
            # Add modifiers
            variations.extend([
                f"best {focus_keyword}",
                f"how to {focus_keyword}",
                f"{focus_keyword} guide",
                f"{focus_keyword} tips",
                f"{focus_keyword} tutorial"
            ])

        return variations[:5]


if __name__ == "__main__":
    # Test the SEO system
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    builder = SEOPromptBuilder()

    # Test daily prompt
    daily_prompt = builder.build_daily_prompt(
        "Artificial Intelligence Breakthrough",
        "New AI model achieves human-level performance"
    )

    print("=" * 70)
    print("DAILY PROMPT TEST")
    print("=" * 70)
    print(daily_prompt[:500] + "...")

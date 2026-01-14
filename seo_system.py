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
                          competitor_insights: Optional[str] = None) -> str:
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

        prompt = f"""
You are an expert SEO content writer and journalist. Write a comprehensive blog post about:

TOPIC: {topic}
CONTEXT: {context}
FOCUS KEYWORD: {focus_keyword}

REQUIREMENTS:

1. SEO META GENERATION (Start your response with these exact lines):
```
SEO_TITLE: [60 character max, compelling title including "{focus_keyword}"]
META_DESCRIPTION: [155-160 character description including "{focus_keyword}"]
FOCUS_KEYWORD: {focus_keyword}
EXCERPT: [2 sentence summary for blog excerpt]
```

2. CONTENT STRUCTURE (HTML format):

H1: {topic}

INTRO (2-3 paragraphs):
- Hook with a compelling fact or question
- Include "{focus_keyword}" naturally in first paragraph
- Set up the article's premise

BODY (3-4 H2 sections):
- Each H2 should include related keywords/LSI terms
- Use short paragraphs (2-3 sentences max)
- Include bulleted lists where appropriate
- Bold key phrases naturally (don't overdo it)
- Include internal link placeholders: [INSERT_INTERNAL_LINK:{focus_keyword}]
- Include external link suggestions: [SUGGEST_EXTERNAL_LINK:authoritative_source]

SEO ELEMENTS:
- Use {focus_keyword} in first 100 words
- Use variations and LSI keywords throughout
- Include at least one question-based H2
- End with a FAQ section with schema markup

FAQ SECTION (Schema.org FAQPage format):
```html
<div class="schema-faq" itemscope itemtype="https://schema.org/FAQPage">
  <h2>Frequently Asked Questions About {focus_keyword}</h2>
  <div itemscope itemprop="mainEntity" itemtype="https://schema.org/Question">
    <h3 itemprop="name">[Question about {topic}]?</h3>
    <div itemscope itemprop="acceptedAnswer" itemtype="https://schema.org/Answer">
      <p itemprop="text">[Detailed answer]</p>
    </div>
  </div>
  <!-- 2-3 more FAQs -->
</div>
```

LENGTH: 800-1200 words

TONE:
- Friendly, engaging, conversational
- Like a knowledgeable friend sharing insights
- Professional but not corporate
- Include some personality and opinions

OUTPUT:
After the content, provide the image generation prompt for the featured image:
```
IMAGE_PROMPT: [Create a professional, modern 16:9 featured image for: {topic}. Style: vibrant but professional. No text overlay.]
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
                           competitor_insights: Optional[str] = None) -> str:
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

        prompt = f"""
You are an expert SEO content writer and senior investigative journalist. Write a comprehensive pillar content article about:

TOPIC: {topic}
CONTEXT: {context}
FOCUS KEYWORD: {focus_keyword}

REQUIREMENTS:

1. SEO META GENERATION (Start your response with these exact lines):
```
SEO_TITLE: [60 character max, compelling title including "{focus_keyword}"]
META_DESCRIPTION: [155-160 character description including "{focus_keyword}"]
FOCUS_KEYWORD: {focus_keyword}
EXCERPT: [3 sentence summary for blog excerpt]
PILLAR_CONTENT: true
```

2. CONTENT STRUCTURE (HTML format):

EXECUTIVE SUMMARY (Styled box):
```html
<div class="wp-block-group has-background" style="background-color:#f0f0f0;padding:25px;border-radius:12px;margin-bottom:30px;border-left:5px solid #0073aa;">
  <h2 style="margin-top:0;">🚀 Executive Summary</h2>
  <p><strong>What you'll learn:</strong></p>
  <ul>
    <li>Key insight 1 about {focus_keyword}</li>
    <li>Key insight 2 with specific data point</li>
    <li>Actionable takeaway for readers</li>
  </ul>
</div>
```

BODY STRUCTURE (Pillar Content - 2000+ words):

SECTION 1: Introduction & Background
- H1: {topic}
- H2: Understanding {focus_keyword}: A Complete Overview
- Cover history, background, and current relevance
- Include statistics and data

SECTION 2: Deep Dive
- H2: How {focus_keyword} Works: The Technical Details
- H3: Key Component 1
- H3: Key Component 2
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
- H2: The Impact of {focus_keyword} on Industry
- H3: Benefits and Advantages
- H3: Challenges and Considerations
- Include case studies or examples

SECTION 4: Future Outlook
- H2: The Future of {focus_keyword}: What to Expect
- Expert predictions and trends
- Timeline graphic placeholder

SECTION 5: Practical Applications
- H2: How to Leverage {focus_keyword} for Success
- Step-by-step guide or actionable tips
- Numbered list for clarity

SECTION 6: FAQs
```html
<div class="schema-faq" itemscope itemtype="https://schema.org/FAQPage">
  <h2>Frequently Asked Questions About {focus_keyword}</h2>
  <!-- 5-7 comprehensive FAQs -->
</div>
```

SEO ELEMENTS:
- Use {focus_keyword} in first 100 words
- Sprinkle LSI keywords throughout
- Include related long-tail keywords in H3s
- At least 5 internal link opportunities
- At least 3 external authority link suggestions
- Optimize for featured snippets (definition boxes, lists)

SEMANTIC HTML:
- Use <section> for major divisions
- Use <figure> and <figcaption> for data visualizations
- Use <blockquote> for expert quotes
- Proper heading hierarchy (H1 → H2 → H3)

LENGTH: 2000-2500 words

TONE:
- Authoritative but accessible
- Data-driven with statistics
- Include expert perspectives
- Forward-looking and analytical

OUTPUT:
After the content, provide:
```
IMAGE_PROMPT: [Create a professional, modern 16:9 featured image for: {topic}. Style: premium business/tech publication. No text overlay. High resolution.]
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

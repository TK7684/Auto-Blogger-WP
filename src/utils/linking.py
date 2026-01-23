"""
Internal Link Resolution Module.
Resolves [INSERT_INTERNAL_LINK:topic] placeholders using existing post data.
"""

import re
import logging
from typing import List, Dict, Optional
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, remove special chars, extra spaces)."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two texts."""
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    return SequenceMatcher(None, norm1, norm2).ratio()


def find_best_match(topic: str, existing_posts: List[Dict[str, str]], min_similarity: float = 0.6) -> Optional[Dict[str, str]]:
    """
    Find the best matching post for a given topic.

    Args:
        topic: The topic to match
        existing_posts: List of posts with 'title' and 'url' keys
        min_similarity: Minimum similarity ratio (0-1) to consider a match

    Returns:
        Best matching post dict or None
    """
    if not existing_posts:
        return None

    best_match = None
    best_score = min_similarity

    for post in existing_posts:
        title = post.get('title', '')

        # 1. Try exact match (case insensitive)
        if topic.lower() == title.lower():
            logger.debug(f"✅ Exact match found: '{topic}' -> '{title}'")
            return post

        # 2. Try contains match (topic is contained in title or vice versa)
        if topic.lower() in title.lower() or title.lower() in topic.lower():
            score = 0.9
            if score > best_score:
                best_score = score
                best_match = post
                logger.debug(f"🔗 Contains match: '{topic}' -> '{title}' (score: {score})")
            continue

        # 3. Try similarity matching for multi-word topics
        topic_words = set(normalize_text(topic).split())
        title_words = set(normalize_text(title).split())

        # Check if any significant words overlap
        overlap = topic_words & title_words
        if overlap and len(overlap) >= min(len(topic_words), len(title_words)):
            score = len(overlap) / max(len(topic_words), len(title_words))
            if score > best_score:
                best_score = score
                best_match = post
                logger.debug(f"🔗 Word overlap match: '{topic}' -> '{title}' (score: {score:.2f})")
            continue

        # 4. Try fuzzy similarity for single words
        similarity = calculate_similarity(topic, title)
        if similarity > best_score:
            best_score = similarity
            best_match = post
            logger.debug(f"🔗 Fuzzy match: '{topic}' -> '{title}' (score: {similarity:.2f})")

    if best_match:
        logger.info(f"✅ Matched '{topic}' to '{best_match['title']}' (similarity: {best_score:.2f})")
    else:
        logger.warning(f"⚠️ No match found for topic: '{topic}' (will use search link)")

    return best_match


def resolve_internal_links(content: str, existing_posts: List[Dict[str, str]]) -> str:
    """
    Resolves [INSERT_INTERNAL_LINK:topic] placeholders using existing post data.

    The function tries multiple matching strategies:
    1. Exact title match (case insensitive)
    2. Contains match (topic in title or title in topic)
    3. Word overlap matching for multi-word topics
    4. Fuzzy similarity matching
    5. Falls back to search-style link if no match found

    Args:
        content: HTML content with [INSERT_INTERNAL_LINK:topic] placeholders
        existing_posts: List of posts with 'title' and 'url' keys

    Returns:
        Content with placeholders replaced by actual HTML links
    """
    if not existing_posts:
        logger.warning("No existing posts provided for internal linking, using search-style links")
        existing_posts = []

    placeholders_found = re.findall(r'\[INSERT_INTERNAL_LINK:([^\]]+)\]', content)

    if placeholders_found:
        logger.info(f"🔗 Found {len(placeholders_found)} internal link placeholders to resolve")
    else:
        return content

    def replacer(match):
        topic = match.group(1).strip()

        # Try to find a matching post
        matched_post = find_best_match(topic, existing_posts)

        if matched_post:
            # Use the actual post URL
            url = matched_post.get('url', '')
            if url:
                return f'<a href="{url}">{topic}</a>'

        # Fallback to search-style link
        search_term = topic.replace(' ', '+')
        return f'<a href="/?s={search_term}">{topic}</a>'

    resolved_content = re.sub(r'\[INSERT_INTERNAL_LINK:([^\]]+)\]', replacer, content)

    # Count how many were resolved to actual posts vs search links
    actual_links = len(re.findall(r'<a href="[^?"]+', resolved_content))
    search_links = len(re.findall(r'<a href="/\?s=', resolved_content))

    logger.info(f"🔗 Resolved: {actual_links} direct links, {search_links} search links")

    return resolved_content


def extract_link_topics(content: str) -> List[str]:
    """Extract all topics from [INSERT_INTERNAL_LINK:topic] placeholders."""
    return re.findall(r'\[INSERT_INTERNAL_LINK:([^\]]+)\]', content)


def clean_remaining_placeholders(content: str) -> str:
    """
    Remove any remaining unresolved placeholders.
    This includes [SUGGEST_EXTERNAL_LINK:...] and any malformed internal link tags.
    """
    # Remove external link suggestions
    content = re.sub(r'\[SUGGEST_EXTERNAL_LINK:[^\]]+\]', '', content)

    # Remove any malformed internal link placeholders
    content = re.sub(r'\[INSERT_INTERNAL_LINK:\s*\]', '', content)

    return content

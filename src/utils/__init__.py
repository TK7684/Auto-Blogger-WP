"""
Utility functions for the Auto-Blogger-WP project.
"""

import re

# ---------------------------------------------------------------------------
# Field alias mapping: common AI-generated key names → canonical snake_case
# field names expected by Pydantic schemas.
#
# After converting raw AI keys to snake_case we apply these aliases so that,
# e.g., a model that returns "CATEGORIES" (→ "categories") is mapped to the
# canonical "suggested_categories" field name.
# ---------------------------------------------------------------------------
FIELD_ALIASES: dict = {
    # content / body
    "article_content": "content",
    "html_content": "content",
    "body": "content",
    "body_content": "content",
    "article": "content",
    "full_content": "content",
    "article_body": "content",
    "post_content": "content",
    # suggested_categories
    "categories": "suggested_categories",
    "category_list": "suggested_categories",
    "wordpress_categories": "suggested_categories",
    "post_categories": "suggested_categories",
    # suggested_tags
    "tags": "suggested_tags",
    "tag_list": "suggested_tags",
    "wordpress_tags": "suggested_tags",
    "post_tags": "suggested_tags",
    # focus_keyword
    "keyword": "focus_keyword",
    "primary_keyword": "focus_keyword",
    "main_keyword": "focus_keyword",
    "target_keyword": "focus_keyword",
    # meta_description
    "description": "meta_description",
    "seo_description": "meta_description",
    "meta_desc": "meta_description",
    # slug
    "url_slug": "slug",
    "post_slug": "slug",
    "url": "slug",
    # in_article_image_prompts
    "image_prompts": "in_article_image_prompts",
    "image_prompts_list": "in_article_image_prompts",
    "in_article_images": "in_article_image_prompts",
    "article_image_prompts": "in_article_image_prompts",
}


def normalize_dict_keys(data: dict) -> dict:
    """
    Normalize dictionary keys to snake_case for Pydantic validation,
    then apply :data:`FIELD_ALIASES` to map common AI-generated key names to
    their canonical Pydantic field names.

    Converts:
    - UPPERCASE_KEYS             → lowercase_keys
    - MixedCaseKeys              → mixed_case_keys
    - already_snake              → already_snake  (unchanged)
    - "CATEGORIES"               → "suggested_categories"  (via alias map)
    - "TAGS"                     → "suggested_tags"         (via alias map)
    - "ARTICLE_CONTENT"/"BODY"   → "content"                (via alias map)

    Examples:
        'SEO_TITLE'          → 'seo_title'
        'META_DESCRIPTION'   → 'meta_description'
        'CATEGORIES'         → 'suggested_categories'
        'FocusKeyword'       → 'focus_keyword'
        'content'            → 'content'

    Args:
        data: Dictionary whose keys should be normalised.

    Returns:
        New dictionary with canonical snake_case keys and the same values.
        Returns the input unchanged if it is not a dict.
    """
    if not isinstance(data, dict):
        return data

    normalized = {}
    for key, value in data.items():
        # Step 1 – insert underscore before a run of capitals followed by a
        #           lower-case letter so we split "ABCDef" → "ABC_Def"
        s1 = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', str(key))
        # Step 2 – insert underscore between a lower-case/digit and an
        #           upper-case letter so "camelCase" → "camel_Case"
        s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1)
        snake_key = s2.lower()
        # Step 3 – apply field alias map (e.g. "categories" → "suggested_categories")
        canonical_key = FIELD_ALIASES.get(snake_key, snake_key)
        normalized[canonical_key] = value

    return normalized

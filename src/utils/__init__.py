"""
Utility functions for the Auto-Blogger-WP project.
"""

import re


def normalize_dict_keys(data: dict) -> dict:
    """
    Normalize dictionary keys to snake_case for Pydantic validation.

    Converts:
    - UPPERCASE_KEYS   → lowercase_keys
    - MixedCaseKeys    → mixed_case_keys
    - already_snake    → already_snake  (unchanged)

    Examples:
        'SEO_TITLE'          → 'seo_title'
        'META_DESCRIPTION'   → 'meta_description'
        'FocusKeyword'       → 'focus_keyword'
        'content'            → 'content'

    Args:
        data: Dictionary whose keys should be normalised.

    Returns:
        New dictionary with snake_case keys and the same values.
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
        normalized[snake_key] = value

    return normalized

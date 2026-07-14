"""
Utility functions for the Auto-Blogger-WP project.
"""

import json
import re


def parse_json_lenient(text: str) -> dict:
    """Parse JSON from LLM output, tolerating markdown fences and leading/trailing noise.

    Z.AI GLM-5.1 (and most LLMs) sometimes wrap JSON in ```json ... ``` fences
    despite schema instructions. This helper strips fences, trims whitespace,
    and falls back to extracting the first {...} block if direct parse fails.
    """
    if text is None:
        raise ValueError("response text is None")
    t = text.strip()
    if not t:
        raise ValueError("response text is empty")
    # Strip BOM (Z.AI may prepend UTF-8 BOM)
    if t.startswith("\ufeff"):
        t = t[1:]
    # Strip ```json / ``` fences
    if t.startswith("```"):
        t = t[3:]
        if t.lower().startswith("json"):
            t = t[4:]
        t = t.lstrip("\r\n")
        if t.endswith("```"):
            t = t[:-3]
        t = t.rstrip()
    # First pass
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # Fallback: first {...} block
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(t[start:end + 1])
        except json.JSONDecodeError:
            pass
    # Fallback 1.5: try json_repair library (handles malformed JSON from LLMs robustly)
    try:
        import json_repair
        repaired_obj = json_repair.loads(t)
        if isinstance(repaired_obj, dict) and repaired_obj:
            return repaired_obj
    except Exception:
        pass
    # Fallback 1.6: try json_repair-style brute-force fix for common LLM issues
    # (unescaped newlines/quotes inside string values)
    repaired = _repair_json_string(t)
    if repaired:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
    # Fallback 2: parse KEY: value text format (Z.AI GLM-5.1 returns this
    # despite structured-output request — prompt template in seo_system.py
    # demonstrates the KV layout, so the model reproduces it)
    kv = _parse_kv_seo_text(text)
    if kv and kv.get("seo_title") and kv.get("content"):
        return kv
    # Re-raise with diagnostic snippet
    raise json.JSONDecodeError(
        f"Failed to parse JSON; len={len(t)}, first 200: {t[:200]!r}",
        t,
        0,
    )


def _repair_json_string(raw: str) -> str | None:
    """Attempt to repair common LLM JSON mistakes and return fixed string (or None).

    Handles:
    - Unescaped newlines inside string values
    - Unescaped quotes inside string values (heuristically)
    - Trailing commas before } or ]
    """
    import re as _re

    text = raw.strip()
    # Quick check: only attempt repair if it looks vaguely JSON-like
    if not text.startswith("{"):
        return None

    # Fix trailing commas before } or ]
    text = _re.sub(r",\s*([}\]])", r"\1", text)

    # Strategy: use a state-machine approach to find unescaped newlines
    # inside string values and escape them.
    result = []
    i = 0
    in_string = False
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\" and i + 1 < len(text):
                # Escaped character — pass through as-is
                result.append(ch)
                result.append(text[i + 1])
                i += 2
                continue
            elif ch == '"':
                result.append(ch)
                in_string = False
                i += 1
                continue
            elif ch == "\n":
                # Unescaped newline inside string — escape it
                result.append("\\n")
                i += 1
                continue
            else:
                result.append(ch)
                i += 1
        else:
            if ch == '"':
                in_string = True
                result.append(ch)
            else:
                result.append(ch)
            i += 1

    repaired = "".join(result)

    # Quick validation: try json.loads on the repaired string
    # If it still fails, try a more aggressive approach: extract content field
    # by finding the longest string value and re-escaping it
    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
        pass

    # More aggressive: try to fix unescaped quotes in string values
    # by finding the content field and re-wrapping it
    content_match = _re.search(r'"content"\s*:\s*"', repaired)
    if content_match:
        # Find the start of the content value
        val_start = content_match.end() - 1  # the opening "
        # Try to find matching close quote by scanning for `,
        # then "suggested_categories" or similar field
        rest = repaired[val_start + 1:]
        # Find where content likely ends — look for `",\n    "next_field":` pattern
        end_match = _re.search(r'(?<!\\)"\s*,\s*\n\s*"', rest)
        if end_match:
            content_raw = rest[:end_match.start()]
            # Re-escape any unescaped quotes in content
            content_fixed = content_raw.replace('\\"', "\x00ESCAPED_QUOTE\x00")
            content_fixed = content_fixed.replace('"', '\\"')
            content_fixed = content_fixed.replace("\x00ESCAPED_QUOTE\x00", '\\"')
            # Also escape newlines
            content_fixed = content_fixed.replace("\n", "\\n")
            new_rest = f'"{content_fixed}"{rest[end_match.start():]}'
            repaired2 = repaired[:val_start] + new_rest
            try:
                json.loads(repaired2)
                return repaired2
            except json.JSONDecodeError:
                pass

    return None


def _parse_kv_seo_text(text: str) -> dict:
    """Parse KEY: value text format that Z.AI GLM-5.1 returns.

    Format:
        SEO_TITLE: ...
        META_DESCRIPTION: ...
        FOCUS_KEYWORD: ...
        EXCERPT: ...
        SUGGESTED_CATEGORIES: a, b, c
        SUGGESTED_TAGS: a, b, c
        <h1>Content...</h1>
        <p>...</p>
    """
    result: dict = {}
    lines = text.split("\n")
    content_start = len(lines)
    field_map = {
        "SEO_TITLE": "seo_title",
        "META_DESCRIPTION": "meta_description",
        "FOCUS_KEYWORD": "focus_keyword",
        "EXCERPT": "excerpt",
        "IMAGE_PROMPT": "image_prompt",
        "SUGGESTED_CATEGORIES": "suggested_categories",
        "SUGGESTED_TAGS": "suggested_tags",
        "CATEGORIES": "suggested_categories",
        "TAGS": "suggested_tags",
    }
    list_fields = {"suggested_categories", "suggested_tags"}
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Content starts at first HTML-looking line or markdown heading
        if stripped.startswith("<h1") or stripped.startswith("<p") or stripped.startswith("# "):
            content_start = i
            break
        # Match KEY: value
        m = re.match(r"^([A-Z_]+)\s*:\s*(.*)$", line)
        if m:
            key_upper = m.group(1)
            val = m.group(2).strip().strip('"').strip("'").strip("[]")
            snake = field_map.get(key_upper)
            if snake:
                if snake in list_fields:
                    result[snake] = [v.strip() for v in val.split(",") if v.strip()]
                else:
                    result[snake] = val
    # Grab content block
    if content_start < len(lines):
        result["content"] = "\n".join(lines[content_start:]).strip()
    # Ensure list fields exist (Pydantic may require them)
    result.setdefault("suggested_categories", [])
    result.setdefault("suggested_tags", [])
    return result

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

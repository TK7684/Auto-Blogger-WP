import re
from typing import List, Dict

def resolve_internal_links(content: str, existing_posts: List[Dict[str, str]]) -> str:
    """
    Resolves [INSERT_INTERNAL_LINK:topic] placeholders using existing post data.
    Attempts to match the topic with titles or slugs.
    """
    if not existing_posts:
        return content

    def replacer(match):
        topic = match.group(1).strip()
        # Search for a match in titles (case insensitive)
        for post in existing_posts:
            if topic.lower() in post['title'].lower():
                return f'<a href="{post["url"]}">{topic}</a>'
        
        # Fallback to search-style link if no direct match found
        return f'<a href="/?s={topic.replace(" ", "+")}">{topic}</a>'

    return re.sub(r'\[INSERT_INTERNAL_LINK:([^\]]+)\]', replacer, content)

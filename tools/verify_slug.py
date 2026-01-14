from main import SEOArticleMetadata
from pydantic import ValidationError

json_output = """
{
    "content": "<h1>Test</h1><p>Content</p>",
    "seo_title": "Test Title",
    "meta_description": "Test Descr",
    "focus_keyword": "test",
    "slug": "test-slug",
    "excerpt": "Test excerpt",
    "suggested_categories": ["Tech"],
    "suggested_tags": ["AI"],
    "in_article_image_prompts": ["Image 1"]
}
"""

try:
    obj = SEOArticleMetadata.model_validate_json(json_output)
    print("✅ Model Validation Passed!")
    print(obj.slug)
except ValidationError as e:
    print(f"❌ Validation Failed: {e}")
except Exception as e:
    print(f"❌ Error: {e}")

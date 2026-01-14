from pydantic import BaseModel, Field
from typing import List

class SEOArticleMetadata(BaseModel):
    content: str = Field(description="The full HTML content of the article")
    seo_title: str = Field(description="SEO-optimized title (max 60 chars)")
    meta_description: str = Field(description="SEO meta description (max 160 chars)")
    focus_keyword: str = Field(description="The primary focus keyword for the article")
    slug: str = Field(description="The URL-friendly slug for the article")
    excerpt: str = Field(description="A short summary of the article")
    suggested_categories: List[str] = Field(description="List of relevant WordPress category names")
    suggested_tags: List[str] = Field(description="List of relevant WordPress tag names")
    in_article_image_prompts: List[str] = Field(description="List of descriptive prompts for images to be placed INSIDE the article content.")

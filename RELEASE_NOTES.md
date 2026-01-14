# Release Notes - v1.1.0 🚀

The **"Gemini 3 Optimization & Research Update"** is here! This release transforms the Auto-Blogging WP application into a state-of-the-art content engine by leveraging the latest AI technology and competitive intelligence.

## Highlights

### 🧠 Gemini 3 Pro & Flash Integration
We've migrated to the brand new `google-genai` (v1.0) SDK.
- **Thinking Mode**: Weekly articles now use "High Thinking Level" for deeper reasoning and superior quality.
- **Structured Outputs**: All AI responses are now strictly validated via Pydantic, eliminating parsing errors.

### 🕵️ Autonomous Research Agent
The new Research Agent analyzes competitive RSS feeds and content gaps on your site to intelligently suggest topics that *matter*. It ensures your blog stays ahead of the curve.

### 🖼️ Next-Gen Image Generation
Integration with **Gemini 3 Pro Image** (Grounded Generation) and **Hugging Face API** as a fallback. 4K high-resolution, 16:9 aspect ratio images are now generated for every post.

### 🔌 Seamless WordPress Integration
Improved term resolution logic allows the AI to suggest categories and tags. The system automatically maps these to your WordPress IDs or creates new tags as needed.

## Installation
Ensure you update your dependencies:
```bash
pip install -r requirements.txt
```

## Config Changes
New environment variables added:
- `HUGGINGFACE_API_KEY`: For alternative image generation.
- `RESEARCH_SOURCES_FILE`: Path to your competitor analysis JSON.

---
*Happy Blogging!*

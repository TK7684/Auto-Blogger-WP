# Release Notes v1.3.0

**Release Date:** 2026-01-20

## 🌟 Highlights

### Multi-Language Auto-Blogging
The system now fully supports **English and Thai** content generation!
- **Random Language Selection**: Automatically switches between English and Thai topics (50/50 mix).
- **Localized Content**: Ensures the AI writes in the correct language for the selected topic.
- **Thai Rescue Topics**: Built-in support for "United SAR K9" and Thai rescue dog topics.

### � Maintenance Agent
A powerful new tool to keep your blog healthy.
- **Run command**: `python main.py maintenance`
- **Auto-Fixes**:
    - Generates missing featured images.
    - Updates outdated content (fact-checking).
    - Fixes placeholder links (`[INSERT_INTERNAL_LINK]`).
    - Formats raw URLs into clickable links.

## � Improvements
- **Centralized Architecture**: Cleaner codebase with all logic moved to `src/`.
- **Enhanced Image Generation**: More reliable image creation using the unified Gemini client.

## 📦 How to Update
1. Pull the latest code.
2. Ensure your `.env` has `GEMINI_API_KEY` and WordPress credentials.
3. Run `python main.py daily` to test the new flow.

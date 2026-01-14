# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-01-06

### Added
- **Gemini 3 Migration**: Fully migrated to `google-genai` SDK (v1.0.0).
- **Research Agent**: New autonomous research module for competitor analysis and content gap detection.
- **Thinking Levels**: Integration of Gemini 3 `thinking_level` parameter (high for weekly, low for daily).
- **Structured Outputs**: Implemented Pydantic models for reliable JSON response parsing.
- **Evergreen Fallbacks**: Robust fallback topics when trend sources are unavailable.
- **Image Generation Enhancements**: Added support for Hugging Face API and Gemini 3 Pro Grounded generation.
- **Unit Testing Suite**: Comprehensive unit tests for `main.py`, `research_agent.py`, and integration tests.

### Changed
- **Optimized Content Generation**: Improved prompts using `SEOPromptBuilder` and structured metadata.
- **Refactored WordPress Integration**: Added support for multiple categories/tags and identity-based term resolution.
- **Performance**: Integrated `requests.Session` for improved network efficiency.

### Fixed
- Fixed RSS date parsing bugs in `research_agent.py`.
- Resolved JSON parsing errors for markdown-wrapped AI responses.
- Cleaned up redundant log files and junk scripts from codebase.

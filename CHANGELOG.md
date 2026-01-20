# Changelog

## [1.3.0] - 2026-01-20

### Added
- **Multi-language Support**: Auto-blogging now supports both English and Thai topics.
- **Maintenance Mode**: New `maintenance` command to audit and fix past posts (add images, fix SEO, clean links).
- **Topics Configuration**: `src/topics.json` for managing topics across languages.
- **Centralized Logic**: Consolidated all core components into `src/` directory.

### Changed
- **Architecture**: Refactored `main.py`, `trend_sources.py`, and `image_generator.py` to use a centralized module structure.
- **Image Generation**: Updated to use the unified `GeminiClient` for better stability and authentication handling.
- **Link Handling**: Auditor now automatically resolves `[INSERT_INTERNAL_LINK:...]` placeholders and fixes raw external URLs.

### Fixed
- **Gemini Client**: Resolved initialization issues where image generator would fail without proper client context.
- **Legacy Files**: Cleaned up obsolete root-level scripts.

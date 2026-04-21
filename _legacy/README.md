# _legacy/

Archived root-level `.py` files dated Apr 17 2026. These were shadow copies of
modules that now live authoritatively in `src/`. Kept per the "Nothing is
Deleted" principle — if a downstream tool accidentally imports from the repo
root, the file history is recoverable here.

Moved 2026-04-21 as part of the Auto-Blogger verify/optimize pass:
- `image_generator.py` — superseded by `src/image_generator.py`
- `research_agent.py`  — superseded by `src/research_agent.py`
- `seo_system.py`      — superseded by `src/seo_system.py`
- `trend_sources.py`   — superseded by `src/trend_sources.py` (rewritten with cadence/type model)
- `yoast_seo.py`       — superseded by `src/yoast_seo.py`

If you are reading this a year from now and nothing references these, delete
the folder.

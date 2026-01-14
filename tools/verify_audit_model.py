from auditor import AuditResult

try:
    res = AuditResult(
        updated_content="<p>New content</p>",
        updated_title="New Clickbait Title",
        fact_check_notes="Checked facts",
        seo_improvements="Improved SEO"
    )
    print("✅ AuditResult model validation passed.")
    print(res.model_dump_json(indent=2))
except Exception as e:
    print(f"❌ AuditResult validation failed: {e}")

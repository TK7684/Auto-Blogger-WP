"""Regression tests for the cross-language focus-keyword bug (post 4843).

Bug: EN crawler topic ("Why You Should Reject Cycle-Based Technical Analysis")
+ TH article → extract_focus_keyword() returned "you reject" (pronouns not in
stopwords), which the LLM then stuffed verbatim into the Thai title/body 13×.

Fixes under test:
1. Stopword expansion (pronouns/function words) — "you reject" class of error
2. Thai-topic branch — Thai topic yields a Thai keyword, never mixed
3. _focus_kw_instruction — EN keyword + Thai article injects the translation
   directive; EN/EN and TH/TH combos inject nothing
"""

import pytest

from src.seo_system import SEOPromptBuilder, _THAI_TOPIC_RE


class TestExtractFocusKeyword:

    def setup_method(self):
        self.builder = SEOPromptBuilder()

    def test_4843_bug_no_more_you_reject(self):
        # exact topic from post 4843
        kw = self.builder.extract_focus_keyword("Why You Should Reject Cycle-Based Technical Analysis")
        assert kw != "you reject"
        assert "you" not in kw.split()
        assert kw  # non-empty

    def test_pronouns_never_in_keyword(self):
        for topic in [
            "Why You Should Reject Cycle-Based Technical Analysis",
            "How I automated my entire smart home with local AI",
            "What They don't tell you about investing in 2026",
        ]:
            kw = self.builder.extract_focus_keyword(topic)
            words = kw.split()
            assert not ({"you", "i", "they", "your", "my", "their", "we", "our"} & set(words)), \
                f"pronoun leaked into keyword {kw!r} from {topic!r}"

    def test_keyword_still_substantive(self):
        kw = self.builder.extract_focus_keyword("Best budget mechanical keyboards for programming")
        assert kw in ("best budget", "budget mechanical", "mechanical keyboards")
        assert len(kw) > 3

    def test_thai_topic_yields_thai_keyword(self):
        kw = self.builder.extract_focus_keyword("แนะนำหูฟังราคาถูกสำหรับฟังเพลงในปี 2026")
        assert _THAI_TOPIC_RE.search(kw)
        assert len(kw) <= 40

    def test_thai_topic_with_english_prefix(self):
        # e.g. "Review: หูฟังบลูทูธที่ดีที่สุด" — the Thai run wins
        kw = self.builder.extract_focus_keyword("Review: หูฟังบลูทูธที่ดีที่สุดปี 2026")
        assert _THAI_TOPIC_RE.search(kw)
        assert kw == "หูฟังบลูทูธที่ดีที่สุดปี"

    def test_english_topic_unchanged_path(self):
        kw = self.builder.extract_focus_keyword("Why You Should Reject Cycle-Based Technical Analysis")
        # first 2 non-stopword words — substantive, no pronouns
        assert kw == "reject cycle-based"


class TestFocusKwInstruction:

    def setup_method(self):
        self.builder = SEOPromptBuilder()

    def test_en_kw_thai_article_injects_translation_directive(self):
        instr = self.builder._focus_kw_instruction("cycle based", "Thai")
        assert "FOCUS KEYWORD TRANSLATION REQUIRED" in instr
        assert "cycle based" in instr
        assert "NEVER insert the English keyword" in instr

    def test_en_kw_english_article_no_directive(self):
        assert self.builder._focus_kw_instruction("cycle based", "English") == ""

    def test_thai_kw_thai_article_no_directive(self):
        assert self.builder._focus_kw_instruction("วิเคราะห์กราฟแบบวัฏจักร", "Thai") == ""

    def test_directive_lands_in_daily_prompt(self):
        p = self.builder.build_daily_prompt(
            "Why You Should Reject Cycle-Based Technical Analysis",
            "ctx", language="Thai")
        assert "FOCUS KEYWORD TRANSLATION REQUIRED" in p

    def test_directive_absent_when_kw_already_thai(self):
        p = self.builder.build_daily_prompt(
            "แนะนำหูฟังราคาถูกสำหรับฟังเพลงในปี 2026",
            "ctx", language="Thai")
        assert "FOCUS KEYWORD TRANSLATION REQUIRED" not in p

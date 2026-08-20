"""Tests for Thai language quality gates (2026-08-20, post-4856 bug class).

Covers:
1. Politeness-particle consistency — mixed ครับ/ค่ะ/คะ in one article is an
   instant bot-tell for native readers.
2. Hallucinated Thai words — 'ลูกหมอก' ("fog child") invented by the model.
3. Thai-text detection.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.verify_published import (
    _is_thai_text,
    _thai_voice_check,
    _thai_word_sanity_check,
)


def _p(*sents):
    return "<p>" + "</p><p>".join(sents) + "</p>"


class TestIsThaiText(unittest.TestCase):
    def test_thai_body_detected(self):
        text = _p("วันนี้เรามาคุยกันเรื่องการเลือกสัตว์เลี้ยงให้เหมาะกับไลฟ์สไตล์ของแต่ละคน",
                  "เพราะแต่ละบ้านมีข้อจำกัดต่างกัน เวลาว่างต่างกัน และงบประมาณก็ต่างกันด้วย")
        self.assertTrue(_is_thai_text(text.replace("<p>", "").replace("</p>", "")))

    def test_english_body_not_thai(self):
        self.assertFalse(_is_thai_text("How to choose the right pet for your lifestyle"))

    def test_short_thai_snippet_not_flagged(self):
        # a lone Thai word in an EN post shouldn't trigger Thai gates
        self.assertFalse(_is_thai_text("Great food at ร้านก๋วยเตี๋ยว last week"))


class TestVoiceConsistency(unittest.TestCase):
    def test_consistent_male_voice_passes(self):
        html = _p("สวัสดีครับ วันนี้มาเล่าเรื่องน้องหมาครับ",
                  "ผมเลี้ยงมาสามปีแล้วครับ สนุกมากครับ")
        passed, detail = _thai_voice_check(html)
        self.assertEqual(passed, "pass")
        self.assertIn("consistent", detail)

    def test_consistent_female_voice_passes(self):
        html = _p("สวัสดีค่ะ วันนี้มาเล่าเรื่องน้องแมวค่ะ",
                  "เราเลี้ยงมาสองปีแล้วค่ะ น่ารักมากค่ะ")
        passed, _ = _thai_voice_check(html)
        self.assertEqual(passed, "pass")

    def test_mixed_particles_fail(self):
        # The post-4856 signature: male body + female FAQ headings
        html = _p("แนะนำแมวครับ เพราะดูแลง่ายกว่าครับ ไม่ต้องพาลงไปไหนครับ",
                  "เหมาะกับคนทำงานเยอะค่ะ",
                  "ลองดูที่เทศบาลก่อนนะคะ",
                  "อีกทีค่ะ")
        passed, detail = _thai_voice_check(html)
        self.assertEqual(passed, "fail")
        self.assertIn("particle mix", detail)

    def test_single_female_quote_in_male_article_ok(self):
        # one quoted line from a female speaker is legitimate Thai writing
        html = _p("เพื่อนบอกว่าน่ารักค่ะ แต่ผมว่าธรรมดาครับ",
                  "ก็ดีครับ ไปต่อครับ")
        passed, _ = _thai_voice_check(html)
        self.assertEqual(passed, "pass")

    def test_no_particles_passes(self):
        passed, detail = _thai_voice_check(_p("น้องหมาน่ารักมาก", "เลือกให้เหมาะกับบ้าน"))
        self.assertEqual(passed, "pass")
        self.assertIn("neutral", detail)


class TestWordSanity(unittest.TestCase):
    def test_normal_thai_passes(self):
        html = _p("เลือกน้องตัวเล็กหรือตัวโตดี", "ดูที่เวลาและพื้นที่ของบ้าน")
        passed, detail = _thai_word_sanity_check(html)
        self.assertEqual(passed, "pass")

    def test_hallucinated_word_fails(self):
        html = _p("เลือกลูกหมอกเล็กหรือโตดีกว่าคะ", "น้องเล็กสอนง่ายกว่า")
        passed, detail = _thai_word_sanity_check(html)
        self.assertEqual(passed, "fail")
        self.assertIn("ลูกหมอก", detail)


if __name__ == "__main__":
    unittest.main()

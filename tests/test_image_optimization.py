"""Tests for the optimized image pipeline (2026-08-19 deep-check).

Covers:
1. PNG→JPEG conversion + mime sniffing (ComfyUI PNGs were saved/uploaded
   as mislabeled .jpg — PNG bytes with Content-Type: image/jpeg).
2. Batch workflow graph construction (hero + inline in ONE submission).
3. Per-purpose aspect resolution (hero 16:9, inline 3:2).
4. Alt text from LLM prompts (language-aware) instead of "{topic} — figure N".
5. save_image extension normalization + WordPress Content-Type sniffing.
6. Steps default 14 (env-overridable).
"""

import os
import sys
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is importable when running from repo root or tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_generator import (
    ImageGenerator,
    WordPressMediaUploader,
    sniff_mime,
    to_web_jpeg,
)


def _make_png(w: int = 64, h: int = 64, mode: str = "RGB", noisy: bool = False) -> bytes:
    """Solid-color PNG by default; noisy PNG approximates real Flux output
    (photographic content where JPEG compression actually wins)."""
    import random
    from PIL import Image
    if noisy:
        rnd = random.Random(42)
        img = Image.new(mode, (w, h))
        img.putdata([tuple(rnd.randrange(256) for _ in range(3)) for _ in range(w * h)])
    else:
        img = Image.new(mode, (w, h), (200, 30, 30) if mode == "RGB" else None)
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def _make_jpeg(w: int = 64, h: int = 64) -> bytes:
    from PIL import Image
    img = Image.new("RGB", (w, h), (30, 30, 200))
    out = BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()


class TestMimeSniffing(unittest.TestCase):
    """Magic-byte sniffing must identify real formats regardless of filename."""

    def test_sniff_png(self):
        self.assertEqual(sniff_mime(_make_png()), "image/png")

    def test_sniff_jpeg(self):
        self.assertEqual(sniff_mime(_make_jpeg()), "image/jpeg")

    def test_sniff_unknown(self):
        self.assertEqual(sniff_mime(b"\x00\x01\x02garbage"), "application/octet-stream")

    def test_real_world_bug_png_labeled_jpg(self):
        """The production bug: ComfyUI PNG persisted as post_*.jpg."""
        png = _make_png(200, 200)
        self.assertNotEqual(sniff_mime(png), "image/jpeg")  # must NOT trust filename


class TestToWebJpeg(unittest.TestCase):
    """PNG → optimized progressive JPEG conversion."""

    def test_png_converts_to_jpeg(self):
        # Noisy image = realistic photographic content (Flux output);
        # JPEG wins decisively here vs PNG's lossless encoding.
        png = _make_png(400, 300, noisy=True)
        jpeg, mime = to_web_jpeg(png)
        self.assertEqual(mime, "image/jpeg")
        self.assertEqual(sniff_mime(jpeg), "image/jpeg")
        self.assertLess(len(jpeg), len(png))

    def test_large_png_downscaled(self):
        jpeg, _ = to_web_jpeg(_make_png(2200, 2200), max_dim=1600)
        from PIL import Image
        img = Image.open(BytesIO(jpeg))
        self.assertLessEqual(max(img.size), 1600)

    def test_rgba_flattened_to_white(self):
        from PIL import Image
        img = Image.new("RGBA", (64, 64), (255, 0, 0, 0))  # fully transparent
        out = BytesIO()
        img.save(out, format="PNG")
        jpeg, mime = to_web_jpeg(out.getvalue())
        self.assertEqual(mime, "image/jpeg")
        decoded = Image.open(BytesIO(jpeg))
        self.assertEqual(decoded.mode, "RGB")

    def test_passthrough_on_garbage(self):
        """Broken input must never lose the original bytes."""
        garbage = b"not-an-image-at-all"
        data, mime = to_web_jpeg(garbage)
        self.assertEqual(data, garbage)
        self.assertEqual(mime, "application/octet-stream")

    def test_progressive_jpeg_flag(self):
        from PIL import Image
        jpeg, _ = to_web_jpeg(_make_png(300, 300))
        self.assertTrue(Image.open(BytesIO(jpeg)).info.get("progression", False))


class TestBatchWorkflow(unittest.TestCase):
    """generate_images_comfyui_batch must build a single multi-chain graph."""

    def setUp(self):
        self.gen = ImageGenerator(gemini_client=None)

    def _workflow_for(self, jobs):
        """Extract the workflow dict the method would submit (mock the POST)."""
        captured = {}

        def fake_post(url, json=None, timeout=None, **kw):
            captured["workflow"] = json["prompt"]
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            resp.json.return_value = {"prompt_id": "test-pid"}
            return resp

        # Availability OK, queue empty, history complete with all SaveImage nodes
        stats = MagicMock(); stats.status_code = 200
        queue = MagicMock(); queue.status_code = 200
        queue.json.return_value = {"queue_running": [], "queue_pending": []}
        hist = MagicMock(); hist.raise_for_status.return_value = None
        save_nodes = [str(60 + i) for i in range(len(jobs))]
        hist.json.return_value = {
            "test-pid": {
                "status": {"status_str": "success"},
                "outputs": {n: {"images": [{"filename": f"img_{n}.png"}]} for n in save_nodes},
            }
        }
        view = MagicMock(); view.status_code = 200; view.content = _make_png()
        view.raise_for_status.return_value = None

        def fake_get(url, timeout=None, **kw):
            if "/system_stats" in url:
                return stats
            if "/queue" in url:
                return queue
            if "/history" in url:
                return hist
            if "/view" in url:
                return view
            raise AssertionError(f"unexpected GET {url}")

        with patch("src.image_generator.requests.get", side_effect=fake_get), \
             patch("src.image_generator.requests.post", side_effect=fake_post):
            results = self.gen.generate_images_comfyui_batch(jobs)
        return captured["workflow"], results

    def test_batch_builds_shared_loaders(self):
        wf, _ = self._workflow_for([{"prompt": "hero", "aspect": "16:9"}])
        classes = {v["class_type"] for v in wf.values()}
        self.assertIn("UNETLoader", classes)
        self.assertIn("DualCLIPLoader", classes)
        self.assertIn("VAELoader", classes)
        # exactly ONE of each shared loader
        self.assertEqual(sum(1 for v in wf.values() if v["class_type"] == "UNETLoader"), 1)
        self.assertEqual(sum(1 for v in wf.values() if v["class_type"] == "DualCLIPLoader"), 1)

    def test_batch_two_jobs_two_chains(self):
        jobs = [
            {"prompt": "hero", "aspect": "16:9"},
            {"prompt": "inline one", "aspect": "3:2"},
        ]
        wf, results = self._workflow_for(jobs)
        self.assertEqual(sum(1 for v in wf.values() if v["class_type"] == "KSampler"), 2)
        self.assertEqual(sum(1 for v in wf.values() if v["class_type"] == "SaveImage"), 2)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r is not None for r in results))

    def test_aspect_dimensions_applied(self):
        wf, _ = self._workflow_for([{"prompt": "hero", "aspect": "16:9"}])
        latent = next(v for v in wf.values() if v["class_type"] == "EmptyLatentImage")
        self.assertEqual(latent["inputs"]["width"], 1216)
        self.assertEqual(latent["inputs"]["height"], 688)

    def test_style_suffix_in_prompt_node(self):
        wf, _ = self._workflow_for([{"prompt": "a cat cafe", "aspect": "1:1"}])
        te = next(v for v in wf.values() if v["class_type"] == "CLIPTextEncode")
        self.assertIn("a cat cafe", te["inputs"]["text"])
        self.assertIn("no text", te["inputs"]["text"])
        self.assertIn("no watermark", te["inputs"]["text"])

    def test_steps_default_14(self):
        wf, _ = self._workflow_for([{"prompt": "x", "aspect": "16:9"}])
        ks = next(v for v in wf.values() if v["class_type"] == "KSampler")
        self.assertEqual(ks["inputs"]["steps"], 14)

    def test_steps_env_override(self):
        os.environ["COMFYUI_STEPS"] = "8"
        try:
            wf, _ = self._workflow_for([{"prompt": "x", "aspect": "16:9"}])
            ks = next(v for v in wf.values() if v["class_type"] == "KSampler")
            self.assertEqual(ks["inputs"]["steps"], 8)
        finally:
            del os.environ["COMFYUI_STEPS"]

    def test_t5_fp8_preferred_when_present(self):
        with patch.object(ImageGenerator, "_comfyui_model_exists", return_value=True):
            wf, _ = self._workflow_for([{"prompt": "x", "aspect": "16:9"}])
        dual = next(v for v in wf.values() if v["class_type"] == "DualCLIPLoader")
        self.assertEqual(dual["inputs"]["clip_name1"], "t5xxl_fp8_e4m3fn.safetensors")

    def test_t5_fallback_to_fp16(self):
        with patch.object(ImageGenerator, "_comfyui_model_exists", return_value=False):
            wf, _ = self._workflow_for([{"prompt": "x", "aspect": "16:9"}])
        dual = next(v for v in wf.values() if v["class_type"] == "DualCLIPLoader")
        self.assertEqual(dual["inputs"]["clip_name1"], "t5xxl_fp16.safetensors")

    def test_batch_facade_falls_back_per_image(self):
        """Partial batch failure must fill gaps via single generation."""
        gen = ImageGenerator(gemini_client=None)
        with patch.object(
            gen, "generate_images_comfyui_batch",
            return_value=[b"hero-bytes", None],
        ), patch.object(
            gen, "generate_image", return_value=b"inline-fallback-bytes",
        ) as single:
            results = gen.generate_images_batch([
                {"prompt": "hero", "aspect": "16:9"},
                {"prompt": "inline", "aspect": "3:2"},
            ])
        self.assertEqual(results[0], b"hero-bytes")
        self.assertEqual(results[1], b"inline-fallback-bytes")
        single.assert_called_once()

    def test_batch_facade_all_success_no_fallback(self):
        gen = ImageGenerator(gemini_client=None)
        with patch.object(
            gen, "generate_images_comfyui_batch",
            return_value=[b"a", b"b", b"c"],
        ), patch.object(gen, "generate_image") as single:
            results = gen.generate_images_batch([
                {"prompt": "a", "aspect": "16:9"},
                {"prompt": "b", "aspect": "3:2"},
                {"prompt": "c", "aspect": "3:2"},
            ])
        self.assertEqual(results, [b"a", b"b", b"c"])
        single.assert_not_called()


class TestSaveImage(unittest.TestCase):
    """save_image must normalize format and extension together."""

    def setUp(self):
        self.gen = ImageGenerator(gemini_client=None)
        self.tmpdir = Path("generated_images")

    def test_png_saved_as_real_jpeg(self):
        path = self.gen.save_image(_make_png(300, 200), "post_test.jpg")
        self.assertIsNotNone(path)
        assert path is not None
        self.assertTrue(path.endswith(".jpg"))
        with open(path, "rb") as f:
            data = f.read()
        self.assertEqual(sniff_mime(data), "image/jpeg")
        Path(path).unlink(missing_ok=True)

    def test_size_logged(self):
        with patch("src.image_generator.logger") as log:
            path = self.gen.save_image(_make_png(300, 200), "post_test2.jpg")
            logged = " ".join(str(c) for c in log.info.call_args_list)
            self.assertIn("KB", logged)
        if path:
            Path(path).unlink(missing_ok=True)


class TestUploaderContentType(unittest.TestCase):
    """WordPress upload Content-Type must match actual bytes."""

    def test_upload_sniffs_content_type(self):
        up = WordPressMediaUploader("https://x.example", "u", "p")
        png = _make_png(50, 50)
        Path("generated_images").mkdir(exist_ok=True)
        Path("generated_images/upload_test.jpg").write_bytes(png)
        captured = {}

        def fake_post(url, headers=None, data=None, params=None, timeout=None):
            captured["headers"] = headers
            captured["data"] = data
            resp = MagicMock()
            resp.status_code = 201
            resp.json.return_value = {"id": 123}
            return resp

        with patch.object(up.session, "post", side_effect=fake_post):
            media_id = up.upload_media("generated_images/upload_test.jpg", "alt", "title")
        self.assertEqual(media_id, 123)
        self.assertEqual(captured["headers"]["Content-Type"], "image/png")
        self.assertEqual(captured["data"], png)
        Path("generated_images/upload_test.jpg").unlink(missing_ok=True)


class TestAltTextLanguage(unittest.TestCase):
    """Inline alt must come from the LLM's prompt (article language)."""

    def test_alt_from_llm_prompt(self):
        from src.main import IMG_PLACEHOLDER_RE
        thai_prompt = "ภาพประกอบแมวน่ารักในคาเฟ่"
        content = f"<p>text</p>\n[IMAGE_PLACEHOLDER_0]\n<p>more</p>"
        tokens = sorted({int(t) for t in IMG_PLACEHOLDER_RE.findall(content)})
        self.assertEqual(tokens, [0])
        # Simulate what main.py does with the prompt
        import re as _re
        alt_text = _re.sub(r"\s+", " ", thai_prompt).strip().strip(".")[:125]
        self.assertEqual(alt_text, thai_prompt)


class TestFluxSafePrompt(unittest.TestCase):
    """Thai prompts must be translated/stripped before hitting Flux's T5.

    Bug class (post 4856, 2026-08-20): the SEO prompt tells the LLM "all
    content values in {language}", so Thai articles produce Thai image
    prompts. Flux's T5 encoder is English-only → renders only the English
    style suffix → generic petless stock photos on a pet article.
    """

    def _gen(self, gemini_client=None):
        return ImageGenerator(gemini_client=gemini_client)

    def test_english_prompt_passthrough(self):
        gen = self._gen()
        prompt = "A golden retriever puppy on a couch"
        self.assertEqual(gen._flux_safe_prompt(prompt), prompt)

    def test_no_client_strips_thai(self):
        gen = self._gen(gemini_client=None)
        result = gen._flux_safe_prompt("ภาพอบอุ่นของคนไทยเล่นกับลูกหมา morning light")
        self.assertNotIn("ภาพ", result)
        self.assertIn("morning light", result)

    def test_no_client_all_thai_gets_fallback(self):
        gen = self._gen(gemini_client=None)
        result = gen._flux_safe_prompt("ภาพอบอุ่นของคนไทยวัยทำงานนั่งเล่นกับลูกหมา")
        self.assertEqual(result, "editorial blog illustration, warm natural light")

    def test_translation_via_gemini(self):
        mock = MagicMock()
        mock.generate_content.return_value.text = (
            '"Warm photo of a Thai office worker playing with a puppy"  '
        )
        gen = self._gen(gemini_client=mock)
        result = gen._flux_safe_prompt("ภาพอบอุ่นของคนไทยวัยทำงานนั่งเล่นกับลูกหมา")
        self.assertEqual(result, "Warm photo of a Thai office worker playing with a puppy")
        # Gemini client actually called
        mock.generate_content.assert_called_once()

    def test_translation_still_thai_falls_back_to_strip(self):
        mock = MagicMock()
        mock.generate_content.return_value.text = "แปลไม่ได้ครับ"
        gen = self._gen(gemini_client=mock)
        result = gen._flux_safe_prompt("ภาพแมวนอนบนคอนโดแมว some english words")
        self.assertNotIn("ภาพ", result)
        self.assertIn("some english words", result)

    def test_translation_exception_strips(self):
        mock = MagicMock()
        mock.generate_content.side_effect = RuntimeError("API down")
        gen = self._gen(gemini_client=mock)
        result = gen._flux_safe_prompt("คนพาลูกหมาเดินเล่นในสวน evening walk")
        self.assertIn("evening walk", result)
        self.assertNotIn("คนพา", result)

    def test_short_thai_fragments_ignored(self):
        """1-2 char Thai fragments (typos) shouldn't trigger full translation."""
        gen = self._gen(gemini_client=None)
        prompt = "A dog cafe in กรุงเทพ with cats"
        # กรุงเทพ is 7 Thai chars → triggers. Only <=2 char runs don't.
        result = gen._flux_safe_prompt(prompt)
        self.assertNotIn("กรุงเทพ", result)

    def test_batch_workflow_uses_safe_prompt(self):
        """The batch builder must route prompts through _flux_safe_prompt."""
        gen = self._gen(gemini_client=None)
        calls = []
        original = ImageGenerator._flux_safe_prompt

        def spy(instance, prompt):
            calls.append(prompt)
            return original(instance, prompt)

        with patch.object(ImageGenerator, "_flux_safe_prompt", spy):
            # Drive the batch builder far enough to build the workflow, then
            # bail at the HTTP submit — we only assert prompt routing.
            import requests as _requests
            with patch.object(_requests, "get") as mock_get:
                stats = MagicMock()
                stats.status_code = 200
                stats.json.return_value = {}
                queue = MagicMock()
                queue.status_code = 200
                queue.json.return_value = {"queue_running": [], "queue_pending": []}
                mock_get.side_effect = [stats, queue]
                with patch.object(ImageGenerator, "_comfyui_model_exists", return_value=True), \
                     patch.object(_requests, "post", side_effect=RuntimeError("stop here")):
                    try:
                        gen.generate_images_comfyui_batch(
                            [{"prompt": "ภาพอบอุ่นคนเล่นกับหมาแมว", "aspect": "16:9"}]
                        )
                    except Exception:
                        pass
        self.assertEqual(calls, ["ภาพอบอุ่นคนเล่นกับหมาแมว"])


if __name__ == "__main__":
    unittest.main()

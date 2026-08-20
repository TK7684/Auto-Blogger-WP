"""Live repair for post 4856 — Thai article whose images were generated from
Thai prompts (invisible to Flux T5 → petless stock portraits).

Regenerates hero + 2 inline images with English prompts (hand-translated
from the Thai alt texts), uploads to WP, swaps srcs + featured media, and
fixes the mixed-language title (EN keyword grafted into Thai).
"""

import os
import sys
import re
import json
import base64
import urllib.request
import datetime as dt

sys.path.insert(0, "/home/tk578/Auto-Blogger-WP")
os.chdir("/home/tk578/Auto-Blogger-WP")

from dotenv import load_dotenv

load_dotenv(override=True)

from src.image_generator import ImageGenerator, WordPressMediaUploader

POST_ID = 4856

# English translations of the Thai alt texts (what Flux should have received)
JOBS = [
    {  # hero (16:9) — replaces media 4853
        "prompt": ("Warm candid lifestyle photo of a Thai office worker relaxing at home "
                   "on a sofa with a golden retriever puppy and a tabby cat, soft morning "
                   "light through curtains, cozy lived-in living room, photorealistic"),
        "aspect": "16:9",
    },
    {  # inline 0 (3:2)
        "prompt": ("Small tidy condominium interior with a fluffy cat lounging on a wooden "
                   "cat tree beside a bright window, plants and soft daylight, photorealistic"),
        "aspect": "3:2",
    },
    {  # inline 1 (3:2)
        "prompt": ("A person walking a small cheerful dog on a leash in a green public park "
                   "in the early evening, golden hour light, Bangkok city skyline in background, "
                   "photorealistic"),
        "aspect": "3:2",
    },
]

NEW_TITLE = "เลี้ยงน้องให้ฟิตกับชีวิต: วิธีเลือกสัตว์เลี้ยงให้เหมาะกับไลฟ์สไตล์"
NEW_META_DESC = ("วิธีเลือกสัตว์เลี้ยงให้เหมาะกับไลฟ์สไตล์ของคุณ เลือกน้องหมาน้องแมวให้ฟิตกับพื้นที่อยู่อาศัย "
                 "เวลาว่าง และงบประมาณ ครบในบทความเดียว")


def wp_request(path, method="GET", body=None):
    auth = base64.b64encode(
        f"{os.environ['WP_USER']}:{os.environ['WP_APP_PASSWORD']}".encode()
    ).decode()
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{os.environ['WP_URL']}/wp-json/wp/v2/{path}",
        data=data,
        method=method,
        headers={
            "Authorization": "Basic " + auth,
            "Content-Type": "application/json",
        },
    )
    return json.load(urllib.request.urlopen(req))


def main():
    print(f"=== Repair post {POST_ID} ===", flush=True)

    # 1. Generate 3 images in one ComfyUI batch (fix included in path)
    gen = ImageGenerator()
    print("Generating 3 images via ComfyUI batch...", flush=True)
    images = gen.generate_images_batch(JOBS)
    ok = sum(1 for r in images if r)
    print(f"Generated {ok}/3", flush=True)
    if ok < 3:
        print("NOT ENOUGH IMAGES — aborting repair", flush=True)
        sys.exit(1)

    # 2. Upload
    uploader = WordPressMediaUploader(
        os.environ["WP_URL"], os.environ["WP_USER"], os.environ["WP_APP_PASSWORD"]
    )
    ts = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    media = []
    alts = [
        "คนไทยวัยทำงานนั่งเล่นกับลูกหมาและแมวในบ้าน แสงเช้าอ่อนๆ บรรยากาศอบอุ่น",
        "คอนโดขนาดเล็กที่จัดระเบียบดี มีแมวนอนเล่นบนคอนโดแมวริมหน้าต่าง",
        "คนพาลูกหมาตัวเล็กเดินเล่นในสวนสาธารณะช่วงเย็น",
    ]
    for i, (img, alt) in enumerate(zip(images, alts)):
        if not img:
            print(f"job {i} has no image bytes — aborting", flush=True)
            sys.exit(1)
        local = gen.save_image(img, f"repair4856_{ts}_{i}.jpg")
        if not local:
            print(f"save_image {i} failed — aborting", flush=True)
            sys.exit(1)
        mid = uploader.upload_media(local, alt, NEW_TITLE if i == 0 else f"สัตว์เลี้ยงให้เหมาะกับไลฟ์สไตล์ — ภาพ {i}")
        if not mid:
            print(f"upload {i} FAILED — aborting", flush=True)
            sys.exit(1)
        src = None
        try:
            m = wp_request(f"media/{mid}")
            src = m.get("source_url")
        except Exception as e:
            print(f"media fetch {mid}: {e}", flush=True)
        media.append((mid, src))
        print(f"uploaded job {i}: media {mid} {src}", flush=True)

    # 3. Fetch post raw content
    post = wp_request(f"posts/{POST_ID}?context=edit")
    content = post["content"]["raw"]

    # Replace hero: featured media = media[0]
    # Replace the two pedpro inline images (keep Shopee product images)
    pedpro_srcs = re.findall(r'src="(https://pedpro\.online/wp-content/uploads/[^"]+)"', content)
    print(f"pedpro inline srcs found: {len(pedpro_srcs)}", flush=True)
    if len(pedpro_srcs) >= 2:
        content = content.replace(pedpro_srcs[0], media[1][1], 1)
        content = content.replace(pedpro_srcs[1], media[2][1], 1)
    # Update alts to the Thai alts above (they were already Thai; refresh anyway)
    # Update figcaptions is cosmetic — skip.

    body = {
        "title": NEW_TITLE,
        "content": content,
        "featured_media": media[0][0],
        "meta": {
            "wpseo_title": NEW_TITLE,
            "wpseo_metadesc": NEW_META_DESC,
            "wpseo_focuskw": "เลือกสัตว์เลี้ยง",
        },
    }
    res = wp_request(f"posts/{POST_ID}", method="POST", body=body)
    print(f"post updated: {res.get('id')} title={res.get('title', {}).get('rendered', '')[:60]}", flush=True)
    print(f"featured_media now: {res.get('featured_media')}", flush=True)

    # 4. Verify
    pub = wp_request(f"posts/{POST_ID}")
    print("VERIFY public:", pub["link"], "| featured:", pub["featured_media"], flush=True)
    imgs = re.findall(r'src="(https://pedpro\.online[^"]+)"', pub["content"]["rendered"])
    for u in imgs:
        print("  img:", u, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

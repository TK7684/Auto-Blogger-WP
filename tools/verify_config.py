import os
from dotenv import load_dotenv

load_dotenv(override=True)

vars_to_check = ["WP_URL", "WP_USER", "WP_APP_PASSWORD", "GEMINI_API_KEY"]

print("--- Config Verification ---")
for var in vars_to_check:
    val = os.environ.get(var)
    if val:
        print(f"{var}: Length={len(val)}, Start={val[:4]}..., End=...{val[-4:]}")
    else:
        print(f"{var}: NOT FOUND")

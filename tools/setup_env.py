import os

def setup_env():
    config = {
        "WP_URL": os.environ.get("WP_URL", ""),
        "WP_USER": os.environ.get("WP_USER", ""),
        "WP_APP_PASSWORD": os.environ.get("WP_APP_PASSWORD", ""),
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
        "TRENDS_GEO": os.environ.get("TRENDS_GEO", "US"),
        "WP_CATEGORY_ID": os.environ.get("WP_CATEGORY_ID", "1")
    }

    with open(".env", "w", encoding="utf-8") as f:
        for key, value in config.items():
            f.write(f"{key}={value}\n")

    print("✅ .env file written successfully and verified.")

if __name__ == "__main__":
    setup_env()

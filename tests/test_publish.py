import os
from dotenv import load_dotenv
import logging

# Set up logging for the test
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def debug_env():
    logging.info(f"Checking environment in: {os.getcwd()}")
    loaded = load_dotenv()
    logging.info(f"load_dotenv() result: {loaded}")
    
    keys_to_check = ["WP_URL", "WP_USER", "WP_APP_PASSWORD", "GEMINI_API_KEY", "TRENDS_GEO"]
    for key in keys_to_check:
        val = os.environ.get(key)
        if val:
            # Mask sensitive info but show it exists
            masked = val[:5] + "..." if len(val) > 5 else "***"
            logging.info(f"✅ {key} is set: {masked}")
        else:
            logging.error(f"❌ {key} is NOT set!")

    from main import generate_content_gemini, post_to_wp
    
    test_topic = "Artificial Intelligence in 2026"
    test_context = "Future trends and human-centered design."
    
    logging.info(f"🚀 Starting Test Flow...")
    article = generate_content_gemini(test_topic, test_context, mode="daily")
    
    if "Failed" in article:
        logging.error(f"❌ Content generation failed: {article}")
    else:
        logging.info("✨ Content generated. Attempting to publish...")
        post_to_wp(f"Test: {test_topic}", article)

if __name__ == "__main__":
    debug_env()

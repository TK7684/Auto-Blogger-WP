"""Debug image data structure."""
from dotenv import load_dotenv
load_dotenv()
import os
import httpx
import json
import base64

api_key = os.environ.get('OPENROUTER_API_KEY')
model = 'google/gemini-3-pro-image-preview'

payload = {
    'model': model,
    'messages': [{'role': 'user', 'content': 'Generate an image: A simple red circle'}],
    'modalities': ['image', 'text'],
}

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json',
    'HTTP-Referer': 'https://pedpro.online',
    'X-Title': 'Auto-Blogger',
}

print('Sending request...')
with httpx.Client(timeout=120.0) as client:
    response = client.post(
        'https://openrouter.ai/api/v1/chat/completions',
        headers=headers,
        json=payload
    )
    data = response.json()
    
    if response.status_code == 200 and 'choices' in data:
        msg = data['choices'][0].get('message', {})
        
        # Debug full message structure
        print(f'Full message structure:')
        print(json.dumps(msg, indent=2, default=str)[:2000])
        
        # Try to extract image
        images = msg.get('images', [])
        if images:
            img = images[0]
            print(f'\nFirst image type: {type(img).__name__}')
            if isinstance(img, dict):
                print(f'Image dict keys: {list(img.keys())}')
                # Try common keys
                for key in ['url', 'data', 'b64_json', 'image_url', 'base64']:
                    if key in img:
                        val = img[key]
                        print(f'  {key}: {type(val).__name__}, len={len(str(val)) if val else 0}')
            elif isinstance(img, str):
                print(f'Image string preview: {img[:100]}...')
    else:
        print(f'Error: {json.dumps(data, indent=2)[:500]}')

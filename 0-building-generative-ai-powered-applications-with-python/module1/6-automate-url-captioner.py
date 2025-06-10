from io import BytesIO

import requests
from bs4 import BeautifulSoup
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

url = "https://en.wikipedia.org/wiki/IBM"

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
img_elements = soup.find_all("img")

captions = {}


for img_element in img_elements:
    img_url = img_element.get("src")

    # Skip if the image is an SVG or too small (likely an icon)
    if "svg" in img_url or "1x1" in img_url:
        continue

    # Correct the URL if it's malformed
    if img_url.startswith("//"):
        img_url = "https:" + img_url
    elif not img_url.startswith("http://") and not img_url.startswith("https://"):
        continue  # Skip URLs that don't start with http:// or https://

    try:
        response = requests.get(img_url)
        raw_image = Image.open(BytesIO(response.content))

        if raw_image.size[0] * raw_image.size[1] < 400:  # Skip very small images
            continue

        raw_image = raw_image.convert("RGB")
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        captions[img_url] = caption
    except Exception as e:
        print(f"Error processing {img_url}: {e}")
        continue

with open("output/captions.txt", "w") as f:
    for img_url, caption in captions.items():
        f.write(f"{img_url}:\n{caption}\n\n")

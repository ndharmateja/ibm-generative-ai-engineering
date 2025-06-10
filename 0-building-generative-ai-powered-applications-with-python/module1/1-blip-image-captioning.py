# Load the pre-trained model and processor
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)


def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption


if __name__ == "__main__":
    # Open the image file
    image = Image.open("data/photographer.png")
    caption = generate_caption(image)

    # Print the generated caption
    print(f"Caption: '{caption}'")

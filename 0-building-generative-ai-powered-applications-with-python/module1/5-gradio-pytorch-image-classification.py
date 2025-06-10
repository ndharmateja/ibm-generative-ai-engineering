import gradio as gr
import requests
import torch
from PIL import Image
from torchvision import transforms

# Import ResNet18
model = torch.hub.load(
    "pytorch/vision:v0.6.0",
    "resnet18",
    weights=True,
).eval()

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")


def predict(inp):
    """
    inp: the input image as a PIL image
    """
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
    return confidences


# Create Gradio interface
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    examples=["data/lion.jpg", "data/cheetah.png"],
).launch(server_name="127.0.0.1", server_port=7860)

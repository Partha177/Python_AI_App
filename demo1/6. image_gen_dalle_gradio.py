# gradio_image_gen.py
import os
import io
import time
import json
from datetime import datetime
from typing import Tuple

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

import requests
from PIL import Image
import gradio as gr


def build_client():
    load_dotenv()
    endpoint = os.getenv("IMAGE_GENERATION_ENDPOINT")
    model_deployment = os.getenv("IMAGE_GENERATION_MODEL_DEPLOYMENT")
    api_version = os.getenv("API_VERSION")
    if not endpoint or not model_deployment or not api_version:
        raise RuntimeError("Missing ENDPOINT / MODEL_DEPLOYMENT / API_VERSION in .env")

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(
            exclude_environment_credential=True,
            exclude_managed_identity_credential=True,
        ),
        "https://cognitiveservices.azure.com/.default",
    )

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
    )
    return client, model_deployment


client, MODEL_DEPLOYMENT = build_client()


def _save_image_from_url(url: str) -> Tuple[Image.Image, str]:
    """
    Download the image and save it under ./images with a unique filename.
    Returns (PIL.Image, file_path).
    """
    # Ensure folder
    out_dir = os.path.join(os.getcwd(), "images")
    os.makedirs(out_dir, exist_ok=True)

    # Unique filename
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_path = os.path.join(out_dir, f"image_{ts}.png")

    # Download + save
    img_bytes = requests.get(url).content
    with open(file_path, "wb") as f:
        f.write(img_bytes)

    # Load to PIL for Gradio preview
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return pil_img, file_path


def generate_image(prompt: str):
    prompt = (prompt or "").strip()
    if not prompt:
        return None, None, "Please enter a prompt."

    try:
        # Call Azure OpenAI Images
        result = client.images.generate(
            model=MODEL_DEPLOYMENT,
            prompt=prompt,
            n=1,  # one image
        )

        # Parse response -> URL, then download
        json_response = json.loads(result.model_dump_json())
        image_url = json_response["data"][0]["url"]
        pil_img, file_path = _save_image_from_url(image_url)

        # Success message
        return pil_img, file_path, f"Image generated and saved: {file_path}"

    except Exception as e:
        return None, None, f"Error: {e}"


with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("# Generate Images with AI (Gradio + Azure OpenAI DALL·E 3)")
    gr.Markdown(
        "Enter a text prompt. The app generates an image using your Azure deployment, "
        "previews it, and lets you download the file."
    )

    prompt = gr.Textbox(label="Prompt", placeholder="e.g., ultra-detailed dragon fruit photo, studio lighting", lines=2)
    btn = gr.Button("Generate", variant="primary")

    with gr.Row():
        img_out = gr.Image(label="Preview", interactive=False)
        file_out = gr.File(label="Download image")

    status = gr.Markdown()

    btn.click(fn=generate_image, inputs=[prompt], outputs=[img_out, file_out, status])

if __name__ == "__main__":
    demo.launch()

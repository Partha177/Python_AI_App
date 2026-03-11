import os
import io
import base64
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from PIL import Image
import gradio as gr

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from openai import AzureOpenAI


def _load_clients():
    load_dotenv()
    endpoint = os.getenv("PROJECT_ENDPOINT")
    deployment = os.getenv("MODEL_DEPLOYMENT")

    if not endpoint or not deployment:
        raise RuntimeError("Missing PROJECT_CONNECTION or MODEL_DEPLOYMENT in .env")

    '''project_client = AIProjectClient(
        credential=DefaultAzureCredential(
            exclude_environment_credential=True,
            exclude_managed_identity_credential=True,
        ),
        endpoint=endpoint,
    )
    openai_client = project_client.get_openai_client(api_version="2024-10-21")'''
    openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-04-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    return openai_client, deployment


openai_client, MODEL_DEPLOYMENT = _load_clients()

SYSTEM_DEFAULT = (
    "You are an AI assistant in a grocery store that sells fruit. "
    "Answer clearly and concisely. If an image is provided, use it to ground your answer, "
    "describing only what is visible."
)


def _to_data_url(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


def answer(question: str, image: Optional[Image.Image], system_msg: str) -> str:
    if not question and image is None:
        return "Please provide a question and/or upload an image."

    content: List[Dict[str, Any]] = []
    if question:
        content.append({"type": "text", "text": question})
    if image is not None:
        content.append({"type": "image_url", "image_url": {"url": _to_data_url(image)}})

    try:
        resp = openai_client.chat.completions.create(
            model=MODEL_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_msg or SYSTEM_DEFAULT},
                {"role": "user", "content": content or [{"type": "text", "text": question}]},
            ],
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("# Develop a Vision-Enabled Chat App (Gradio + Azure AI Foundry)")
    gr.Markdown("Upload an image and ask a question. Uses your configured model deployment via Azure AI Projects.")

    with gr.Row():
        image = gr.Image(type="pil", label="Upload image (optional)")
        with gr.Column():
            system_box = gr.Textbox(label="System message", value=SYSTEM_DEFAULT, lines=3)
            question = gr.Textbox(label="Your question", placeholder="e.g., What fruit is this? Is it ripe?")

    ask = gr.Button("Ask")
    output = gr.Markdown()

    ask.click(answer, inputs=[question, image, system_box], outputs=[output])

if __name__ == "__main__":
    demo.launch()

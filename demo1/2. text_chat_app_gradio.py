# gradio_text_chat.py
import os
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
import gradio as gr

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from openai import AzureOpenAI


SYSTEM_DEFAULT = (
    "You are a helpful AI assistant that answers questions clearly and concisely."
)

def _build_client():
    load_dotenv()
    endpoint = os.getenv("PROJECT_ENDPOINT")
    deployment = os.getenv("MODEL_DEPLOYMENT")
    if not endpoint or not deployment:
        raise RuntimeError(
            "Missing PROJECT_ENDPOINT or MODEL_DEPLOYMENT in .env"
        )
    '''project_client = AIProjectClient(
        credential=DefaultAzureCredential(
            exclude_environment_credential=True,
            exclude_managed_identity_credential=True
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

openai_client, MODEL_DEPLOYMENT = _build_client()

def _chat(messages: List[Dict[str, Any]]) -> str:
    resp = openai_client.chat.completions.create(
        model=MODEL_DEPLOYMENT,
        messages=messages
    )
    return resp.choices[0].message.content

def send_message(user_msg: str,
                 #chat_history: List[Tuple[str, str]],
                 chat_history: List[Dict[str, str]],
                 system_msg: str,
                 messages_state: List[Dict[str, Any]]):
    '''if not user_msg.strip():
        return gr.update(), messages_state'''
    if not user_msg.strip():
        return chat_history, messages_state

    # Append user turn
    messages_state.append({"role": "user", "content": user_msg})

    try:
        assistant = _chat(messages_state)
    except Exception as e:
        assistant = f"Error: {e}"

    # Append assistant turn
    messages_state.append({"role": "assistant", "content": assistant})
    #chat_history = chat_history + [(user_msg, assistant)]
    chat_history = chat_history + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant},
    ]
    return chat_history, messages_state

def reset_chat(system_msg: str):
    # (re)seed message list with system message
    return [], [{"role": "system", "content": system_msg or SYSTEM_DEFAULT}]

with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("# Generative AI Chat (Gradio + Azure AI Foundry)")
    gr.Markdown("Type a question and get an answer using your configured Azure model deployment.")

    system_box = gr.Textbox(
        label="System message (optional)",
        value=SYSTEM_DEFAULT,
        lines=3
    )

    chatbot = gr.Chatbot(height=420)
    user_box = gr.Textbox(
        label="Your message",
        placeholder="Ask anything…",
    )

    messages_state = gr.State(value=[{"role": "system", "content": SYSTEM_DEFAULT}])

    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear")

    # Wire actions
    send_btn.click(
        send_message,
        inputs=[user_box, chatbot, system_box, messages_state],
        outputs=[chatbot, messages_state]
    )
    user_box.submit(
        send_message,
        inputs=[user_box, chatbot, system_box, messages_state],
        outputs=[chatbot, messages_state]
    )
    clear_btn.click(
        reset_chat,
        inputs=[system_box],
        outputs=[chatbot, messages_state]
    )

if __name__ == "__main__":
    demo.launch(share=True)

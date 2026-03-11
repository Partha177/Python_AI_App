import os, time, warnings
from typing import List, Tuple, Optional

import gradio as gr
from dotenv import load_dotenv
from openai import AzureOpenAI

# Silence Assistants deprecation warnings so the UI stays clean
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ----- Load env -----
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview").strip()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o").strip()
ASSISTANT_ID_ENV = os.getenv("ASSISTANT_ID", "").strip()
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "KiddoTutor").strip()

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
    raise RuntimeError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env")

# ----- Client -----
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=API_VERSION,
)

INSTRUCTIONS = (
    "You are a fun and friendly teacher for kids. Use simple language, emojis, and stories "
    "to explain concepts. Make learning interactive with questions, rhymes, and quizzes. "
    "Keep the tone playful and supportive.\n\n"
    "Important: Reply only when the user’s query is about kids learning, teaching, quizzes, "
    "rhymes, or fun knowledge. For all other topics, politely say: "
    "\"I can only help with kids’ learning.\""
)

def ensure_assistant() -> str:
    """Reuse ASSISTANT_ID from .env if valid; otherwise create a new assistant."""
    if ASSISTANT_ID_ENV:
        try:
            client.beta.assistants.retrieve(assistant_id=ASSISTANT_ID_ENV)
            print(f"Using existing assistant: {ASSISTANT_ID_ENV}")
            return ASSISTANT_ID_ENV
        except Exception as e:
            print(f"Could not retrieve ASSISTANT_ID={ASSISTANT_ID_ENV}: {e} — creating a new one.")

    a = client.beta.assistants.create(
        model=DEPLOYMENT,
        name=ASSISTANT_NAME,
        instructions=INSTRUCTIONS,
        tools=[],  # no tools in this minimal demo
    )
    print(f"Created assistant: {a.id}")
    return a.id

ASSISTANT_ID = ensure_assistant()

def new_thread() -> str:
    t = client.beta.threads.create()
    return t.id

def add_user_message(thread_id: str, text: str) -> None:
    client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=text
    )

def run_and_get_reply(thread_id: str) -> str:
    run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=ASSISTANT_ID)
    while run.status in ("queued", "in_progress", "cancelling"):
        time.sleep(0.4)
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)

    if run.status != "completed":
        return f"[Run status: {run.status}]"

    # Fetch newest assistant message
    msgs = client.beta.threads.messages.list(thread_id=thread_id, order="desc", limit=6)
    for m in msgs.data:
        if m.role == "assistant":
            parts = []
            for c in m.content:
                if c.type == "text":
                    parts.append(c.text.value)
            if parts:
                return "\n".join(parts)
    return "(No assistant reply found.)"

# ---------------- Gradio UI ----------------
with gr.Blocks(fill_height=True, theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 👶 KiddoTutor (Azure OpenAI Assistants) — Gradio Chat")

    chat = gr.Chatbot(height=480, show_copy_button=True)
    msg = gr.Textbox(placeholder="Ask something for kids… (type and press Enter)", label="Message")
    with gr.Row():
        clear_btn = gr.Button("New Chat", variant="secondary")
        info = gr.Markdown("", elem_id="info")

    # Per-session state: each browser tab gets its own thread
    thread_state = gr.State(value=None)  # will hold thread_id string

    def respond(user_msg: str, history: List[Tuple[str, str]], thread_id: Optional[str]):
        if not user_msg.strip():
            return gr.update(value=""), history, thread_id

        if thread_id is None:
            thread_id = new_thread()

        add_user_message(thread_id, user_msg)
        reply = run_and_get_reply(thread_id)
        history = history + [(user_msg, reply)]
        return gr.update(value=""), history, thread_id

    def new_chat():
        # Start a fresh thread and clear the chat window
        return new_thread(), []

    msg.submit(respond, [msg, chat, thread_state], [msg, chat, thread_state])
    clear_btn.click(new_chat, outputs=[thread_state, chat])

if __name__ == "__main__":
    # Visit http://127.0.0.1:7860
    demo.launch()

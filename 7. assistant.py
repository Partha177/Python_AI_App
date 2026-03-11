import os, time, warnings
from dotenv import load_dotenv
from openai import AzureOpenAI

# Hide Assistants deprecation warnings so the REPL looks clean
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
ASSISTANT_ID = os.getenv("ASSISTANT_ID", "").strip()
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "KiddoTutor")

INSTRUCTIONS = (
    "You are a fun and friendly teacher for kids. Use simple language, emojis, and stories to explain concepts. "
    "Make learning interactive with questions, rhymes, and quizzes. Keep the tone playful and supportive.\n\n"
    "Important: Reply only when the user’s query is about kids learning, teaching, quizzes, rhymes, or fun knowledge. "
    "For all other topics, politely say: \"I can only help with kids’ learning.\""
)

def ensure_assistant() -> str:
    """Use ASSISTANT_ID if provided; otherwise create a new one."""
    if ASSISTANT_ID:
        try:
            client.beta.assistants.retrieve(assistant_id=ASSISTANT_ID)
            print(f"Using existing assistant: {ASSISTANT_ID}")
            return ASSISTANT_ID
        except Exception as e:
            print(f"Could not retrieve ASSISTANT_ID={ASSISTANT_ID}: {e} — creating a new one.")

    a = client.beta.assistants.create(
        model=DEPLOYMENT,
        name=ASSISTANT_NAME,
        instructions=INSTRUCTIONS,
        tools=[],  # no tools in this minimal demo
    )
    print(f"Created assistant: {a.id}")
    return a.id

def new_thread_id() -> str:
    t = client.beta.threads.create()
    return t.id

def add_user_message(thread_id: str, text: str) -> None:
    client.beta.threads.messages.create(thread_id=thread_id, role="user", content=text)

def run_and_print(thread_id: str, assistant_id: str) -> None:
    run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)
    while run.status in ("queued", "in_progress", "cancelling"):
        time.sleep(0.5)
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)

    if run.status != "completed":
        print(f"[Run status: {run.status}]")
        return

    # Fetch latest messages and print newest assistant reply
    msgs = client.beta.threads.messages.list(thread_id=thread_id, order="desc", limit=10)
    for m in msgs.data:
        if m.role == "assistant":
            # A message can have multiple content parts; print text parts
            out = []
            for part in m.content:
                if part.type == "text":
                    out.append(part.text.value)
            if out:
                print("\nAssistant:", "\n".join(out), "\n")
            break

def main():
    assistant_id = ensure_assistant()
    thread_id = new_thread_id()

    print("Interactive KiddoTutor. Type 'exit' to quit.\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("exit", "quit", "q"):
            break

        add_user_message(thread_id, user)
        run_and_print(thread_id, assistant_id)

if __name__ == "__main__":
    main()

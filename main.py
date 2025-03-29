from fastapi import FastAPI, Request, HTTPException
import redis
from langchain.memory import ConversationBufferMemory
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os

app = FastAPI()

# Initialize Redis
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

# Check if model exists and print debug info
model_path = "./Meta-Llama-3-8B-Instruct-v2.Q2_K.gguf"
if not os.path.exists(model_path):
    raise Exception(f"Model file not found at {model_path}")
print(f"Model file size: {os.path.getsize(model_path)} bytes")

# Initialize Llama model with n_gpu_layers
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
try:
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.75,
        max_tokens=2000,
        n_ctx=1024,
        n_gpu_layers=8,  # Adjusted for RTX 4050 with 8GB VRAM
        callback_manager=callback_manager,
        verbose=True,
    )
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise


def get_memory(session_id: str):
    """Retrieve or create memory for a given session."""
    history = redis_client.get(session_id)
    memory = ConversationBufferMemory(return_messages=True)
    if history:
        memory.chat_memory.messages = eval(history)  # Convert back to list
    return memory


@app.get("/")
async def hello():
    return "<h1> Vikas</h1>"


@app.post("/chat")
async def chat(request: Request, user_input: str, session_id: str):
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")

    memory = get_memory(session_id)

    # Use Llama model for response
    response = llm(memory.chat_memory.messages + [user_input])

    # Update and store conversation history
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(response)
    redis_client.set(session_id, str(memory.chat_memory.messages))

    return {"response": response, "session_id": session_id}

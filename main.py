from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import traceback
import json
import redis
from langchain.memory import ConversationBufferMemory
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
import re
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Redis setup
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

# Model setup
model_path = "./Llama-3.2-8B-Instruct-Q4_K_M.gguf"
if not os.path.exists(model_path):
    raise Exception(f"Model file not found at {model_path}")

# System prompt using format the model might better recognize
SYSTEM_PROMPT = """<s>[INST] <<SYS>>
You are a helpful, direct assistant. Reply to the user's questions directly and concisely.
Do not add meta-commentary, role-playing suggestions, or conversation simulations.
Do not include options for users to choose.
Respond as if you are having a direct conversation.
<</SYS>>
"""

# Use stop sequences to prevent generating beyond what we want
STOP_SEQUENCES = [
    "User:",
    "[/INST]",
    "</s>",
    "<s>",
    "Options:",
    "**Options",
    "Please choose",
    "Please respond",
    "You can respond",
    "Role-play",
]

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.5,  # Lower temperature for more deterministic output
    max_tokens=200,
    n_ctx=2048,
    n_gpu_layers=8,
    stop=STOP_SEQUENCES,  # Use stop sequences to prevent unwanted continuations
    callback_manager=callback_manager,
    verbose=False,
)


# Helper functions
def get_memory(session_id: str) -> ConversationBufferMemory:
    """Retrieve or create memory for a given session"""
    memory = ConversationBufferMemory(return_messages=True)
    try:
        history = redis_client.get(session_id)
        if history:
            messages = json.loads(history)
            for msg in messages[-6:]:  # Keep last 6 messages to avoid context overflow
                if msg["type"] == "human":
                    memory.chat_memory.add_user_message(msg["content"])
                elif msg["type"] == "ai":
                    memory.chat_memory.add_ai_message(msg["content"])
    except Exception as e:
        print(f"Error loading memory: {str(e)}")
    return memory


def save_memory(session_id: str, memory: ConversationBufferMemory):
    """Save memory to Redis"""
    messages = []
    for msg in memory.chat_memory.messages:
        if msg.type == "human":
            messages.append({"type": "human", "content": msg.content})
        elif msg.type == "ai":
            messages.append({"type": "ai", "content": msg.content})
    redis_client.setex(session_id, 86400, json.dumps(messages))  # Expire after 1 day


def clean_llm_response(response: str) -> str:
    """
    Intelligently clean LLM responses by identifying and preserving the actual content
    while removing formatting artifacts, meta-text, and unwanted symbols.
    """
    original_response = response
    
    # First pass: Remove any XML-like tags or Llama tag markers
    response = re.sub(r'<[^>]+>', '', response)
    response = re.sub(r'\[INST\].*?\[/INST\]', '', response, flags=re.DOTALL)
    response = re.sub(r'<<SYS>>.*?<</SYS>>', '', response, flags=re.DOTALL)
    
    # Second pass: Cut off at any meta-text indicators
    cutoff_indicators = [
        "Options:", "**Options", "User:", "Please choose", 
        "Please respond", "You can respond", "Role-play",
        "|", "---", "**A)", "**B)", "A)", "B)",
        "<userStyle>", "<userExamples>"
    ]
    
    lowest_index = len(response)
    for indicator in cutoff_indicators:
        index = response.find(indicator)
        if index != -1 and index < lowest_index:
            lowest_index = index
    
    if lowest_index < len(response):
        response = response[:lowest_index]
    
    # Third pass: Clean specific patterns
    # Remove any Assistant: prefix
    response = re.sub(r'^Assistant:\s*', '', response.strip())
    
    # Remove meta instructions
    response = re.sub(r'Let\'s continue the conversation.*', '', response, flags=re.DOTALL)
    response = re.sub(r'Here is the continuation.*', '', response, flags=re.DOTALL)
    
    # Fourth pass: Remove common symbols that appear at the end
    response = re.sub(r'<s>$', '', response)
    response = re.sub(r'</s>$', '', response)
    response = re.sub(r'\[/INST\]$', '', response)
    
    # Fifth pass: Clean trailing symbols or special markers
    while re.search(r'[<>\[\]{}()/\\|]$', response):
        response = re.sub(r'[<>\[\]{}()/\\|]$', '', response)
    
    # Final cleanup of whitespace and ensure there's no trailing symbols
    response = response.strip()
    
    # Check if we have a valid response
    # If the response is too short or empty, try to extract the meaningful part
    if not response or len(response) < 10:
        # Look for the first proper sentence in the original response
        sentences = re.findall(r'[A-Z][^.!?]*[.!?]', original_response)
        if sentences:
            # Find the first sentence that seems like a real response (not meta-text)
            for sentence in sentences:
                # Skip sentences that are likely part of instructions
                if not any(indicator in sentence for indicator in cutoff_indicators):
                    if len(sentence.strip()) > 5:
                        return sentence.strip()
        
        # If we still don't have a valid response, check if there are any meaningful phrases
        phrases = re.findall(r'[A-Za-z][^,.!?]{3,}[,.!?]', original_response)
        if phrases:
            best_phrase = max(phrases, key=len)
            if len(best_phrase.strip()) > 10:
                return best_phrase.strip()
        
        # If all else fails, check if we have any sequence of words that seems reasonable
        words_match = re.search(r'[A-Za-z\s]{15,}', original_response)
        if words_match:
            return words_match.group(0).strip()
            
        # As a last resort, return a varied default response based on a hash of the original
        # This ensures we don't always return the same default message
        default_responses = [
            "I'm here to help. What would you like to talk about?",
            "How can I assist you today?",
            "I'd be happy to help with your question.",
            "Feel free to ask me anything.",
            "What can I help you with?"
        ]
        
        # Use a hash of the original response to select a default response
        # This ensures some variety while being deterministic for the same input
        response_hash = sum(ord(c) for c in original_response) % len(default_responses)
        return default_responses[response_hash]
    
    return response

# API endpoints
class ChatRequest(BaseModel):
    user_input: str
    session_id: str


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Validate input
        if not request.session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")

        # Get/create memory
        memory = get_memory(request.session_id)
        memory.chat_memory.add_user_message(request.user_input)

        # Format prompt using Llama 2/3 instruction format
        prompt = SYSTEM_PROMPT

        # Add the most recent history (to keep context manageable)
        recent_messages = (
            memory.chat_memory.messages[-5:]
            if len(memory.chat_memory.messages) > 5
            else memory.chat_memory.messages
        )

        for msg in recent_messages:
            if msg.type == "human":
                prompt += f"{msg.content} [/INST] "
            elif msg.type == "ai":
                prompt += f"{msg.content} </s><s>[INST] "

        # Add final prompt marker if the last message was from the user
        if recent_messages and recent_messages[-1].type == "human":
            prompt += " [/INST] "

        # Generate response
        raw_response = llm(prompt)
        print(f"Raw response: {raw_response}")  # Debugging

        # Clean up response
        response = clean_llm_response(raw_response)
        print(f"Cleaned response: {response}")  # Debugging

        # Save assistant's response to memory
        memory.chat_memory.add_ai_message(response)

        # Save updated memory
        save_memory(request.session_id, memory)

        return {"response": response, "session_id": request.session_id}

    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Error: {str(e)}\n{traceback_str}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug-response")
async def debug_response(request: ChatRequest):
    """Endpoint for debugging the raw responses from the model"""
    try:
        # Format prompt the same way as the chat endpoint
        prompt = SYSTEM_PROMPT + f"{request.user_input} [/INST] "

        # Get raw response without cleaning
        raw_response = llm(prompt)

        # Also get the cleaned version
        cleaned_response = clean_llm_response(raw_response)

        return {
            "raw_response": raw_response,
            "cleaned_response": cleaned_response,
            "prompt": prompt,
        }
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Error: {str(e)}\n{traceback_str}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clear-memory/{session_id}")
async def clear_memory(session_id: str):
    try:
        redis_client.delete(session_id)
        return {"message": f"Memory cleared for session {session_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

import os
import time
import openai
from dotenv import load_dotenv
from prompts_logger import PromptLogger
import tiktoken
from google import genai
from anthropic import Anthropic
from anthropic.types import TextBlock

# Load environment variables from .env file
load_dotenv()

# Configure the primary OpenAI client for official OpenAI models
api_key = os.getenv("OPENAI_API_KEY")

ENCODING = "o200k_base"   # works for GPT-4o / GPT-5 family
MAX_TOKENS = 280000

def crop_to_tokens(text: str, max_tokens: int = MAX_TOKENS) -> str:
    enc = tiktoken.get_encoding(ENCODING)
    tokens = enc.encode(text)
    return enc.decode(tokens[:max_tokens])

if not api_key:
    # This is now a warning, as you might only want to use Ollama
    print("Warning: OPENAI_API_KEY not found. Only local Ollama models will be available.")
    primary_openai_client = None
else:
    primary_openai_client = openai.OpenAI(api_key=api_key)

# List of all available models, both OpenAI and Ollama
AVAILABLE_MODELS = {
    # OpenAI Models
    "gpt-5.4-mini": "gpt-5.4-mini-2026-03-17", 
    "gpt-5.2": "gpt-5.2-2025-12-11", 
    "gpt-5.2-pro": "gpt-5.2-pro-2025-12-11", 
    "gpt-5": "gpt-5-2025-08-07",
    "gpt-5.1":"gpt-5.1-2025-11-13",
    "gpt-5-mini": "gpt-5-mini-2025-08-07",
    "gpt-5-nano": "gpt-5-nano-2025-08-07",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
    "gemini-3.1-pro": "models/gemini-3.1-pro-preview",
    "claude-fable-5": "claude-fable-5", 
    "gpt-5.6-sol": "gpt-5.6-sol",
    # Ollama Models
    # "gemma3": "gemma3:27b-it-q4_K_M",
    "qwen3": "qwen3:8b",
}

def get_llm_response(model_name: str, prompt: str, temperature: float = 0.7, json_mode: bool = False, logger: PromptLogger = None):
    """
    Gets a response from the specified LLM using an OpenAI-compatible API.
    Routes requests to either the official OpenAI API or a local Ollama server.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' is not available. Choose from {list(AVAILABLE_MODELS.keys())}")

    actual_model_name = AVAILABLE_MODELS[model_name]
    is_ollama_model = ":" in actual_model_name
    is_claude = False
    is_gemini = False
   
    
    prompt = crop_to_tokens(prompt)
    # --- MODIFIED: Select the correct client configuration ---
    if "claude" in model_name:
        client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
        is_claude = True

    elif "gemini" in model_name:
        client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))
        is_gemini = True

    elif is_ollama_model:
        # For Ollama models, create a client pointed at the local server
        client = openai.OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # Required by the library, but Ollama doesn't use it
        )
    else:
        # For official models, use the primary pre-configured client
        if not primary_openai_client:
            raise ValueError(f"Cannot use model '{model_name}' because OPENAI_API_KEY is not configured.")
        client = primary_openai_client
    
    max_retries = 5
    retry_delay = 10  # seconds

    for attempt in range(max_retries):
        try:
            # --- MODIFIED: This API call is now IDENTICAL for both services ---
            response_format = {"type": "json_object"} if json_mode else {"type": "text"}
            mime_type = "application/json" if json_mode else "text/plain"
            if is_claude:
                try:
                    response = client.messages.create(
                        model=actual_model_name,
                        max_tokens=8192,   # required — caps the OUTPUT length
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                    )
                    response_content = ""
                    
                    for block in response.content:
                        if isinstance(block, TextBlock):
                            response_content += block.text

                except Exception as e:
                    response = client.messages.create(
                        model="claude-sonnet-4-6",  # fallback model
                        max_tokens=8192,   # required — caps the OUTPUT length
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                    )
                    response_content = ""
                    
                    for block in response.content:
                        if isinstance(block, TextBlock):
                            response_content += block.text

                
                # response_content = response.content[0].text.strip()
                
                if response.stop_reason == "max_tokens":
                    # response was cut off — either raise max_tokens or continue the generation
                    print("Warning: response truncated")
                    continued = client.messages.create(
                        model=actual_model_name,
                        max_tokens=8192,
                        messages=[
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response.content},  # the truncated output
                            {"role": "user", "content": "continue"},
                        ],
                    )
                    for block in response.content:
                        if isinstance(block, TextBlock):
                            response_content += block.text
                    # response_content = continued.content[0].text.strip()

            elif is_gemini:
                response = client.models.generate_content(
                    model=actual_model_name,
                    contents=prompt,
                    config={
                        "response_mime_type": mime_type
                    }
                )
                response_content = response.text.strip()

            else:
                completion = client.chat.completions.create(
                    model=actual_model_name,
                    messages=[
                        # {"role": "system", "content": "You are a helpful assistant participating in a simulation."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=response_format,
                )
                response_content = completion.choices[0].message.content.strip()

            if logger:
                logger.log(prompt, response_content)

            return response_content

        # This error handling now works for both OpenAI and Ollama through the compatible API
        except openai.RateLimitError as e:
            print(f"Rate limit exceeded. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)
        except openai.APIError as e:
            print(f"API error: {e}. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Retrying... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)

    raise Exception("Failed to get a response from the LLM after multiple retries.")

def get_batch_responses_from_llm(
        msg,
        client,
        model,
        system_message,
        print_debug=False,
        msg_history=None,
        temperature=0.75,
        n_responses=1,
):
    if msg_history is None:
        msg_history = []

    if 'gpt' in model:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=4096,
            n=n_responses,
            stop=None,
            seed=0,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    else:
        content, new_msg_history = [], []
        for _ in range(n_responses):
            c, hist = get_llm_response(
                msg,
                client,
                model,
                system_message,
                print_debug=False,
                msg_history=None,
                temperature=temperature,
            )
            content.append(c)
            new_msg_history.append(hist)

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history
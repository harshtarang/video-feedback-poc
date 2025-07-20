# https://platform.openai.com/docs/api-reference/responses/create

import os
from openai import OpenAI

from constants import OPENAI_SEED

# Model pricing in dollars per token (input cost per token, output cost per token)
MODEL_PRICING = {
    "gpt-4o-mini": (0.50 / 1_000_000, 1.50 / 1_000_000),
    "gpt-4o": (5.00 / 1_000_000, 15.00 / 1_000_000),
    "gpt-4-turbo": (10.00 / 1_000_000, 30.00 / 1_000_000),
    "gpt-3.5-turbo": (0.50 / 1_000_000, 1.50 / 1_000_000),
}

MAX_COMPLETION_TOKENS = int(os.getenv("OPENAI_MAX_COMPLETION_TOKENS", 4095))
print(f"Using max completion tokens: {MAX_COMPLETION_TOKENS}")

def call_openai(
    query, temperature=0, top_p=1, max_completion_tokens=MAX_COMPLETION_TOKENS, response_format=None, provider="OpenAI"
):

    if provider == "Gemini":
        model = "gemini-2.0-flash"  # Default model for Gemini
        base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        api_key = os.getenv("GEMINI_API_KEY")
    else:  # OpenAI
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        base_url = os.getenv("OPENAI_BASE_URL", None)
        api_key = os.getenv("OPENAI_API_KEY")
        
    client = OpenAI(api_key=api_key, base_url=base_url)
    if response_format != None:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_completion_tokens,
            response_format=response_format,
            # seed=OPENAI_SEED,  # Set seed for reproducibility
        )
    else:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_completion_tokens,
            # seed=OPENAI_SEED,  # Set seed for reproducibility
        )
    # Calculate and print cost
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    
    # Only show cost for OpenAI calls
    if provider == "OpenAI":
        if model in MODEL_PRICING:
            input_price, output_price = MODEL_PRICING[model]
            cost = (prompt_tokens * input_price) + (completion_tokens * output_price)
            print(f"OpenAI API call cost: ${cost:.6f} (model: {model}, prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens})")
        else:
            print(f"OpenAI API call: model {model} not found in pricing dictionary. Token usage: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}")
    
    return completion.choices[0].message.content


def get_asr_transcription(audio_file, keyword_list, provider="OpenAI"):

    # Whisper is only available on OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    with open(audio_file, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="en",
            prompt=keyword_list,
            response_format="verbose_json",  # JSON format includes timestamps
            timestamp_granularities=["word"],
            include=["logprobs"],
        )

    # Calculate ASR cost: $0.006 per minute
    cost_per_minute = 0.006
    cost = (transcription.duration / 60) * cost_per_minute
    print(f"OpenAI ASR transcription cost: ${cost:.6f} (duration: {transcription.duration:.2f} seconds)")
    
    return transcription


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    response = call_openai("Tell me a joke")

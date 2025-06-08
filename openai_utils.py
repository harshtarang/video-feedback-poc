# https://platform.openai.com/docs/api-reference/responses/create

import os
from openai import OpenAI

MAX_COMPLETION_TOKENS = os.getenv("OPENAI_MAX_COMPLETION_TOKENS", 512)

def call_openai(
    query, temperature=0, top_p=1, max_completion_tokens=MAX_COMPLETION_TOKENS, response_format=None
):

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL", None)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base_url)
    if response_format != None:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_completion_tokens,
            response_format=response_format,
        )
    else:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_completion_tokens,
        )
    # print(completion.choices[0].message.content)
    return completion.choices[0].message.content


def get_asr_transcription(audio_file, keyword_list):

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    audio_file = open(audio_file, "rb")

    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        # response_format="text",
        language="en",
        prompt=keyword_list,
        response_format="verbose_json",  # JSON format includes timestamps
        timestamp_granularities=["word"],
        # normalize=False,
        include=["logprobs"],
    )
    return transcription


if __name__ == "__main__":
    pass

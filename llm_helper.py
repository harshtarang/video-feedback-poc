import json
import os
from openai_utils import call_openai, get_asr_transcription
from prompt_templates import AUDIO_PROMPT, TEXT_PROMPT, TEXT_QUALITY_PROMPT


def speech_to_text(audio_file, transcription_file, keyword_list, file_name="good"):
    # Whisper is only available on OpenAI
    if os.getenv("USE_CACHE_TRANSCRIPTION") == "True" and os.path.exists(f"cache/transcriptions/transcription_{file_name}.json"):
        print("*************************** USING CACHED TRANSCRIPTION **********************************")
        # Load cached transcription
        cached_transcription_file = f"cache/transcriptions/transcription_{file_name}.json"
        with open(cached_transcription_file) as f:
            transcription_dict = json.load(f)
    else:
        transcription = get_asr_transcription(audio_file, keyword_list)
        transcription_dict = transcription.to_dict()
    with open(transcription_file, "w") as f:
        json.dump(transcription_dict, f)
    print("*************************** TRANSCRIPTION DONE AND SAVED IN " + transcription_file + " **********************************")

def prompt_for_audio(aug_trans_file, output_file, model="OpenAI", audio_prompt=None):

    with open(aug_trans_file) as f:
        aug_trans = f.read()
        
    print("************************ GETTING AUDIO FEEDBACK AND SAVING IN " + output_file + " **********************************")
    
    # Use provided prompt or default
    prompt_template = audio_prompt if audio_prompt else AUDIO_PROMPT
    aud_prompt = prompt_template.format(augmented_transcription = aug_trans)
    
    print(aud_prompt)
    print("*"*50)
    aud_fb = call_openai(aud_prompt, provider=model)
    print("*"*50)
    print(aud_fb)
    
    return aud_fb
        
def prompt_for_text(gt_file, timed_transcription, model="OpenAI", text_prompt=None, quality_prompt=None):

    with open(gt_file) as f:
        gt = f.read()
        
    with open(timed_transcription) as f:
        if timed_transcription.endswith("json"):
            transcription = json.load(f)
            trans = transcription["text"]
        else:
            trans = f.read()
    
    
    print("************************ GETTING TEXT FEEDBACK **********************************")
    
    # Use provided prompts or defaults
    text_prompt_template = text_prompt if text_prompt else TEXT_PROMPT
    quality_prompt_template = quality_prompt if quality_prompt else TEXT_QUALITY_PROMPT
    
    prompt = text_prompt_template.format(ground_truth_text = gt, asr_transcription = trans)
    print(prompt)
    print("*"*50)
    correctness_feedback = call_openai(prompt, provider=model)
    
    prompt = quality_prompt_template.format(ground_truth_text = gt, asr_transcription = trans)
    quality_feedback = call_openai(prompt, provider=model)

    return correctness_feedback, quality_feedback
    

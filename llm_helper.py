import json
import os
from constants import MAX_NUM_LLM_CALLS
from openai_utils import call_openai, get_asr_transcription
from prompt_templates import AUDIO_PROMPT, AUDIO_PROMPT_V2, DISFLUENCY_PROMPT, TEXT_PROMPT, TEXT_PROMPT_V1, TEXT_QUALITY_PROMPT
from utilities import extract_json_from_llm_feedback


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

    prompt_template = audio_prompt if audio_prompt else AUDIO_PROMPT_V2
    aud_prompt = prompt_template.format(augmented_transcription = aug_trans)
    print("*"*50)
    
    is_format_correct = "False"
    num_llm_calls = 0
    
    while num_llm_calls < MAX_NUM_LLM_CALLS and is_format_correct != "Success":
        num_llm_calls += 1
        aud_fb = call_openai(aud_prompt, provider=model)
        print("*"*50)
        print(aud_fb)
        is_format_correct = extract_json_from_llm_feedback(aud_fb, "audio")
        
    
    with open(output_file, "w") as f:
        f.write(aud_fb)
        
def prompt_for_text(gt_file, timed_transcription, output_file, model="OpenAI", text_prompt=None):

    with open(gt_file) as f:
        gt = f.read()
        
    with open(timed_transcription) as f:
        if timed_transcription.endswith("json"):
            transcription = json.load(f)
            trans = transcription["text"]
        else:
            trans = f.read()
    
    
    print("************************ GETTING TEXT FEEDBACK AND SAVING IN " + output_file + " **********************************")
    
    # Use provided prompts or defaults
    text_prompt_template = text_prompt if text_prompt else TEXT_PROMPT_V1
    
    prompt = text_prompt_template.format(ground_truth_text = gt, asr_transcription = trans)

    is_format_correct = "False"
    num_llm_calls = 0
    
    while num_llm_calls < MAX_NUM_LLM_CALLS and is_format_correct != "Success":
        num_llm_calls += 1
        text_fb = call_openai(prompt, provider=model)
        is_format_correct = extract_json_from_llm_feedback(text_fb, "text")
        
    
    with open(output_file, "w") as f:
        f.write(text_fb)
    
def prompt_for_quality(gt_file, timed_transcription, output_file, model="OpenAI", quality_prompt=None):

    with open(gt_file) as f:
        gt = f.read()
        
    with open(timed_transcription) as f:
        if timed_transcription.endswith("json"):
            transcription = json.load(f)
            trans = transcription["text"]
        else:
            trans = f.read()
    
    
    print("************************ GETTING QUALITY FEEDBACK AND SAVING IN " + output_file + " **********************************")
    
    # Use provided prompts or defaults
    quality_prompt_template = quality_prompt if quality_prompt else DISFLUENCY_PROMPT
    
    prompt = quality_prompt_template.format(ground_truth_text = gt, asr_transcription = trans)

    is_format_correct = "False"
    num_llm_calls = 0
    
    while num_llm_calls < MAX_NUM_LLM_CALLS and is_format_correct != "Success":
        num_llm_calls += 1
        text_fb = call_openai(prompt, provider=model)
        is_format_correct = extract_json_from_llm_feedback(text_fb, "quality")
        
    
    with open(output_file, "w") as f:
        f.write(text_fb)
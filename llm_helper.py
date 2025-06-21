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

def prompt_for_audio(aug_trans_file, output_file, model="OpenAI"):

    with open(aug_trans_file) as f:
        aug_trans = f.read()
        
    print("************************ GETTING AUDIO FEEDBACK AND SAVING IN " + output_file + " **********************************")
    aud_prompt = AUDIO_PROMPT.format(augmented_transcription = aug_trans)
    print(aud_prompt)
    print("*"*50)
    aud_fb = call_openai(aud_prompt, provider=model)
    print("*"*50)
    print(aud_fb)
    # with open(output_file, "w") as f:
    #     f.write(aud_fb)
    
    return aud_fb
        
def prompt_for_text(gt_file, timed_transcription, model="OpenAI"):

    with open(gt_file) as f:
        gt = f.read()
    with open(timed_transcription) as f:
        transcription = json.load(f)
    trans = transcription["text"]
    
    
    print("************************ GETTING TEXT FEEDBACK **********************************")
    prompt = TEXT_PROMPT.format(ground_truth_text = gt, asr_transcription = trans)
    print(prompt)
    print("*"*50)
    correctness_feedback = call_openai(prompt, provider=model)
    prompt = TEXT_QUALITY_PROMPT.format(ground_truth_text = gt, asr_transcription = trans)
    quality_feedback = call_openai(prompt, provider=model)

    return correctness_feedback, quality_feedback
    

import whisper
import string

MODEL = whisper.load_model("base.en")
FILLER_SET = set(["uhh", "uh", "uhm", "uhmm", "hmm", "umm", "mm", "hm", "mhmm"])

def is_overlap(a_start, a_end, b_start, b_end):
    return not (b_end <= a_start or b_start >= a_end)

def merge_non_overlapping(first, second):
    # Assumes both lists are sorted by start_time
    result = first[:]
    i = 0  # Pointer for first list

    for b_start, b_end, b_text in second:
        overlap_found = False
        # Since both are sorted, we can move i forward as needed
        while i < len(first) and first[i][1] <= b_start:
            i += 1
        # Now check from current i if overlap exists
        j = i
        while j < len(first) and first[j][0] < b_end:
            a_start, a_end, _ = first[j]
            if is_overlap(a_start, a_end, b_start, b_end):
                overlap_found = True
                break
            j += 1
        if not overlap_found:
            result.append([b_start, b_end, b_text])

    # Sort the result by start time again
    result.sort(key=lambda x: x[0])
    return result

    
def get_filler_whisper_os(audio_f):
    filler_list = []
    result = MODEL.transcribe(audio_f, initial_prompt= "umm, uh, uhh, uhm, uhmm, hmm, mm, hm, mhmm, let me think like, hmm... Okay, here's what I'm, like, thinking.")
    
    for seg in result["segments"]:
        word_set = set([x.lower().strip().strip(string.punctuation) for x in seg["text"].split()])
        if len(word_set.intersection(FILLER_SET)) > 0:
            curr_filler_list = [seg["start"], seg["end"], seg["text"]]
            filler_list.append(curr_filler_list)
    return filler_list
    
def get_filler_whisper_api(sent_level_transcript):   
    filler_list = []
    with open(sent_level_transcript) as f:
        text = f.readlines()
    for t in text:
        time_text = t.split("seconds>")
        time_interval = time_text[0].split(">: <")[1].strip().split("-")
        curr_text = time_text[1].strip()
        word_set = set([x.lower().strip() for x in curr_text.split()])
        if len(word_set.intersection(FILLER_SET)) > 0:
            curr_filler_list = [int(time_interval[0].strip()), int(time_interval[1].strip()), curr_text]
            filler_list.append(curr_filler_list)
    return filler_list
            
def filler_aggregation(audio_f, sent_level_transcript, output_file):
    filler_list_os = get_filler_whisper_os(audio_f)
    
    filler_list_api = get_filler_whisper_api(sent_level_transcript)

    
    final_filler_list = merge_non_overlapping(filler_list_os, filler_list_api)
    
    for fll in final_filler_list:
        print("FILLER WORD DETECTED BETWEEN " + str(fll[0]) + " and " + str(fll[1]) + " seconds")
        print("*"*50)
        print(fll[2])
        print("-"*50)
    
    if final_filler_list != []:
        with open(output_file, "w") as f:
            for fll in final_filler_list:
                f.write("FILLER WORD DETECTED BETWEEN " + str(fll[0]) + " and " + str(fll[1]) + " seconds\n")
                f.write("*"*50 + "\n")
                f.write(fll[2] + "\n")
                f.write("="*50 + "\n")
        
if __name__ == "__main__":
    filler_aggregation(audio_f = "artifacts/audios/audio_Udiliv_Rank5.wav", sent_level_transcript = "artifacts/transcriptions/sentence_level_transcription_Udiliv_Rank5.txt", output_file = "1.txt")
    
            
    
    

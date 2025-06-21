import os

import pandas as pd

from utilities import longest_zero_sequence, read_feat

def filler_detection(timed_transcription, silence_file, window_interval = 0.02):

    silence = read_feat(silence_file, is_pitch = False)
    
    '''
    #timed_transcription = os.path.join("artifacts/transcriptions", "transcription_" + os.path.splitext(input_vid_file)[0] + ".json")
    with open(timed_transcription) as f:
        transcription = json.load(f)
    duration = transcription["duration"]
    
    text = modify_sentence(transcription["text"])#.replace("-", " ").replace(",", " ").replace("  ", " ")
    print(text)
    text_words = text.split()
    timed_transcription_df = pd.DataFrame(transcription["words"])
    
    timed_transcription_df["word"] = timed_transcription_df["word"].apply(lambda x: modify_sentence(x).split())
    timed_transcription_df = timed_transcription_df.explode("word").reset_index(drop=True)
    timed_transcription_df = timed_transcription_df.fillna("")
    '''
    timed_transcription_df = pd.read_csv(timed_transcription)
    sentences_set_with_disfluencies = []
    times = []
    for idx in range(2, len(timed_transcription_df) - 1):
        prev_word_end_time = timed_transcription_df["end"][idx]
        next_word_st_time = timed_transcription_df["start"][idx + 1]
        
        if timed_transcription_df["syllables_per_sec"].iloc[idx] >= 9 or timed_transcription_df["syllables_per_sec"].iloc[idx + 1] >= 9:
            continue
        if next_word_st_time != prev_word_end_time:
            ffile_st = int(prev_word_end_time / window_interval)
            ffile_en = max(int(next_word_st_time / window_interval), ffile_st + 1)
            feat_word = silence[ffile_st: ffile_en]
            non_silent_non_word_duration = longest_zero_sequence(feat_word) * window_interval ## thresh 200ms = 0.2s
            if non_silent_non_word_duration > 0.35: ## Minimum filler word duration 350ms
                time_of_disfluency = int(0.5*(prev_word_end_time + next_word_st_time)) 
                curr_sentence_list = list(set(timed_transcription_df["sentence_id"][idx: idx + 2]))
                if sentences_set_with_disfluencies == [] or curr_sentence_list[0] > sentences_set_with_disfluencies[-1][-1] + 1:
                    sentences_set_with_disfluencies.append(curr_sentence_list)
                    times.append([time_of_disfluency])
                elif curr_sentence_list[0] == sentences_set_with_disfluencies[-1][-1] or curr_sentence_list[0] == sentences_set_with_disfluencies[-1][-1] + 1:
                    sentences_set_with_disfluencies[-1][-1] = curr_sentence_list[-1]
                    times[-1].append(time_of_disfluency)
                
                print("**** FILLER WORD ***")
                print(non_silent_non_word_duration, idx, prev_word_end_time, next_word_st_time, timed_transcription_df["word"].iloc[idx], timed_transcription_df["word"].iloc[idx + 1], sentences_set_with_disfluencies, times)
    
    filler_feedback = ""      
    for idx in range(len(times)):
        req_sentence = ""
        time_str = (", ").join([str(x) + " sec" for x in times[idx]])
        for sid in sentences_set_with_disfluencies[idx]:
            #print(sid)
            req_sentence += (" ").join(timed_transcription_df[timed_transcription_df["sentence_id"] == sid]["word"]) + ". "
        filler_feedback += "Disfluencies found around " + time_str + "\n" + req_sentence + "\n"

    return filler_feedback
        
def speaking_rate_feedback(timed_transcription):
    word_df = pd.read_csv(timed_transcription)
    max_sentence_id = word_df["sentence_id"].iloc[-1]
    for sid in range(1, max_sentence_id):
        dfr = word_df[word_df["sentence_id"] == sid]
        for idx in range(len(dfr)):
            if dfr["SYLL_RATE"].isna.all():
                continue
            df_high = dfr[dfr["SYLL_RATE"].str.contains("HIGH_PACE")]
            df_low = dfr[dfr["SYLL_RATE"].str.contains("LOW_PACE")] 
               
            
if __name__ == "__main__":
    input_vid_file = "Dupha_AV4.mp4"
    timed_transcription = os.path.join("artifacts/audio_feats", "word_level_feats_" + os.path.splitext(input_vid_file)[0] + ".csv")
    silence = os.path.join("artifacts/audio_feats", "silence_" + os.path.splitext(input_vid_file)[0] + ".txt")
    nonllm_feedback_file = os.path.join("artifacts/feedback", "nonllm_feedback_" + os.path.splitext(input_vid_file)[0] + ".txt")
    filler_detection(timed_transcription, silence, nonllm_feedback_file, window_interval = 0.02)
    #speaking_rate_feedback(timed_transcription)
## Include punctuation in word. If no punctuation in +/- K neighbor words but significant pause > 0.3, then flag. Also flag if no pause but there is , or . or ?
## Flag if Energy/Pitch of keywords below 50% energy in sentence. Get list of 50% energy and match them to keyword set. make sure to account for spell mistakes
## Should start and end of sentence have threshold? So if your sentences follow bell curve energy than flag. Only flag is sentence is big enough
## "," and "and" if silence is high around and then ignore
## What other features? Decide and run on 5/10 samples.

## Audio based filler word detector or Assembely AI API

## Text features (prompt gpt)
## Spelling mistakes -- ask the user to say the word clearly, Flag False starts, Filler words, Content analysis

## Compute Energy/Pitch/Syllables per second every at least 4 words or 5 syllables
## In silence computation, include pause between end time stamps
## https://imotions.com/blog/learning/research-fundamentals/voice-analysis-the-complete-pocket-guide/
##     Intonation scores are derived from pitch standard deviation and have a typical range of 0.4 – 1.6. Values below 0.4 indicate monotonous speech Values above 1.6 indicate lively speaking


import json
import string
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
pd.options.mode.chained_assignment = None
from utilities import count_syllables, find_boundaries, modify_sentence, read_feat

GLOBAL_LOW_PITCH_THRESH = 80
GLOBAL_HIGH_SYLL_RATE = 5
GLOBAL_LOW_SYLL_RATE = 3
GLOBAL_MAX_PAUSE = 2

HIGH_SYLL_RATE = 6
LOW_SYLL_RATE = 2

    
def compute_word_level_audio_feat(st, en, ffile, feat_type = None, silence_threshold = None, window_interval = 0.02):
    ffile_st = int(st / window_interval)
    ffile_en = max(int(en / window_interval), ffile_st + 1)
    feat_word = ffile[ffile_st: ffile_en]
    if feat_type == "pitch":
        feat_word = [x for x in feat_word if x != 0]
        if len(feat_word) == 0:
            return 0
    if feat_type == "energy":
        feat_word = [x for x in feat_word if x > silence_threshold]
        if len(feat_word) == 0:
            return silence_threshold
    '''
    if feat_type == "silence":
        print(feat_word)
        return list(feat_word).count(1) * window_interval ## thresh 200ms = 0.2s
    '''    
    return np.mean(feat_word)


def word_level_feat_computation(timed_transcription, pitch_file, energy_file, silence_file, output_file_word_level_feat, output_file_aat, output_file_tt, window_interval = 0.02):
    
    pitch = read_feat(pitch_file, is_pitch = True)
    energy = read_feat(energy_file, is_pitch = False)
    silence = read_feat(silence_file, is_pitch = False)
    
    ### Compute rms threshold for low volume
    HINRGDIV=50 	       #Dead silence frames ratio (100)
    LOX=1.5		       #Multiply low (min) energy estimate by this
    PERCENTTHRESH=0.07	
    sortrms=list(energy)					
    sortrms.sort()
    hinrg=sortrms[int(0.9*len(sortrms))]
    
    dead=0                                      #Find and discount any dead silence (< 90%_max/50) frames
    for ss in sortrms:
        if ss < hinrg/HINRGDIV: dead += 1	#Count # of dead silence frames
    print("dead =",dead)
    lonrg=sortrms[int(0.1*len(sortrms))+dead]	#Find 10th (+ dead) percentile

    speechthresh = 2*(hinrg-lonrg)*PERCENTTHRESH + LOX*lonrg	#= % of max-min
    silence_thresh = (hinrg-lonrg)*PERCENTTHRESH + LOX*lonrg
    print("lonrg =",lonrg)
    print("hinrg =",hinrg)
    print("speechthresh =",speechthresh)
    ### End of computing threshold
      
    with open(timed_transcription) as f:
        transcription = json.load(f)
    duration = transcription["duration"]
    
    text = modify_sentence(transcription["text"])#.replace("-", " ").replace(",", " ").replace("  ", " ")
    print(text)
    text_words = text.split()
    word_df = pd.DataFrame(transcription["words"])
    
    word_df["word"] = word_df["word"].apply(lambda x: modify_sentence(x).split())
    word_df = word_df.explode("word").reset_index(drop=True)
    word_df = word_df.fillna("")
    word_df["num_syllables"] = word_df["word"].apply(lambda x: count_syllables(x))
    
    ### PAUSE ANNOTATION ###
    word_df["PAUSE"] = [" "]*len(word_df)
    word_df["PAUSE_DURATION"] = [None]*len(word_df)
    pause_boundaries = find_boundaries(silence)
    for idx in range(len(pause_boundaries)):
        end_second = pause_boundaries[idx][1] * window_interval
        start_second = pause_boundaries[idx][0] * window_interval
        duration = round(end_second - start_second, 2)
        dfr = word_df[word_df["start"] < start_second]
        #print(pause_boundaries[idx], pause_boundaries[idx][0], start_second, end_second, len(dfr))
        #print(list(dfr["word"]))
        if len(dfr) > 0:
            #print(dfr.index)
            #print(Counter(list(dfr.index)))
            req_idx = list(dfr.index)[-1]
            #print(dfr["start"].iloc[-1], dfr["end"].iloc[-1], dfr["word"].iloc[-1])
            word_df["PAUSE"].iloc[req_idx] = " <PAUSE: " + str(duration) + "s> "
            word_df["PAUSE_DURATION"].iloc[req_idx] = duration
        #print("="*50)
        
    ### PHRASE LEVEL (3 WORDS OR 5 SYLLABLES) PITCH ENERGY PACE COMPUTATION. TIMESTAMPS ARE MORE ACCURATE THAN WORD LEVEL ####   
    word_df["PHRASE_PITCH"] = [None]*len(word_df)
    word_df["PHRASE_ENERGY"] = [None]*len(word_df)
    word_df["PHRASE_NUM_SYLL"] = [None]*len(word_df)
    word_df["PHRASE_DURATION"] = [None]*len(word_df)
    word_df["PHRASE_SYLL_RATE"] = [None]*len(word_df)
    st_idx = 0
    en_idx = 0
    num_words = 0
    num_syllables = 0
    while st_idx < len(word_df):
        while (num_words < 3 or num_syllables < 5) and en_idx < len(word_df):
            num_words += 1
            num_syllables += word_df["num_syllables"].iloc[en_idx] 
            en_idx += 1
        start_time = word_df["start"].iloc[st_idx]
        end_time = word_df["end"].iloc[en_idx - 1]
        
        phrase_duration = end_time - start_time
        phrase_pitch = compute_word_level_audio_feat(start_time, end_time, pitch, feat_type = "pitch")
        phrase_energy = compute_word_level_audio_feat(start_time, end_time, energy, feat_type = "energy", silence_threshold = silence_thresh)
        phrase_syllable = sum(word_df["num_syllables"].iloc[st_idx: en_idx])
        
        word_df["PHRASE_PITCH"].iloc[st_idx: en_idx] = phrase_pitch
        word_df["PHRASE_ENERGY"].iloc[st_idx: en_idx] = phrase_energy
        word_df["PHRASE_NUM_SYLL"].iloc[st_idx: en_idx] = phrase_syllable
        word_df["PHRASE_DURATION"].iloc[st_idx: en_idx] = phrase_duration
        word_df["PHRASE_SYLL_RATE"].iloc[st_idx: en_idx] = phrase_syllable/phrase_duration
        st_idx = en_idx
        num_words = 0
        num_syllables = 0
        
    HIGH_PITCH_THRESH = np.percentile(word_df["PHRASE_PITCH"], 90)
    LOW_PITCH_THRESH = np.percentile(word_df["PHRASE_PITCH"], 10)
    HIGH_ENERGY_THRESH = np.percentile(word_df["PHRASE_ENERGY"], 90)
    LOW_ENERGY_THRESH = np.percentile(word_df["PHRASE_ENERGY"], 10)
    
    print("HIGH PITCH THRESHOLD: %.2f Hz" % (HIGH_PITCH_THRESH))
    print("LOW PITCH THRESHOLD: %.2f Hz" % (LOW_PITCH_THRESH))
    print("HIGH ENERGY THRESHOLD: %.2f" % (HIGH_ENERGY_THRESH))
    print("LOW ENERGY THRESHOLD: %.2f" % (LOW_ENERGY_THRESH))
    print("*"*50)
    word_df["PITCH"] = word_df["PHRASE_PITCH"].apply(lambda x: "HIGHER_PITCH: " + str(int(x)) + "Hz> " if x > HIGH_PITCH_THRESH else "LOWER_PITCH: " + str(int(x)) + "Hz> " if x < LOW_PITCH_THRESH else None)
    word_df["ENERGY"] = word_df["PHRASE_ENERGY"].apply(lambda x: "HIGHER_ENERGY: " + str(x) + "> " if x > HIGH_ENERGY_THRESH else "LOWER_ENERGY: " + str(x) + "> " if x < LOW_ENERGY_THRESH else None)
    
    word_df["SYLL_RATE"] = word_df["PHRASE_SYLL_RATE"].apply(lambda x: "HIGH_PACE: " + str(round(x,1)) + " syl/s> " if x > HIGH_SYLL_RATE else "LOW_PACE: " + str(round(x, 1)) + " syl/sec> " if x < LOW_SYLL_RATE else None)
    text_idx = 0
    sentence_id = [0]*len(word_df)
    sentence_length = np.array([0]*len(word_df))
    sentence_length_st = 0
    curr_sentence_id = 0
    curr_sentence_length = 0
    
    for idx in range(len(word_df)):
        curr_sentence_length += 1
        curr_word = word_df["word"].iloc[idx] # Timed transcription word
        curr_sentence_word = text_words[text_idx]  ## transcription word with punctuation
        punctuated = False
        if curr_sentence_word[-1] in string.punctuation:
            punctuated = True
            punct = curr_sentence_word[-1]
            curr_sentence_word = curr_sentence_word[:-1]
            #print(curr_sentence_word, punct)
        #print(curr_word, curr_sentence_word, text_idx, curr_sentence_id)
        
        if curr_word.strip() == "":
            sentence_id[idx] = curr_sentence_id      
        elif curr_sentence_word == curr_word:
            sentence_id[idx] = curr_sentence_id
            text_idx += 1
        else:
            print("!!!!! ERROR !!!! CHECK HERE: WORDS DONT ALING IN TIMED AND UNTIMED TRANSCRIPTIONS")
            print(curr_word, curr_sentence_word, text_idx, curr_sentence_id)
            break
        if punctuated and punct in [".", "?"] and curr_sentence_length > 1:
            curr_sentence_id += 1
            sentence_length[sentence_length_st: sentence_length_st + curr_sentence_length] = curr_sentence_length
            sentence_length_st = sentence_length_st + curr_sentence_length
            curr_sentence_length = 0
    
            
    print("="*50)
    
    try:
        assert(curr_sentence_id == len(text.split(". ")))
    except:
        print("WARNING: THERE COULD BE ERROR IN LOGIC. NUMBER OF SENTENCES IN TEXT AND TOTAL FROM SENTENCE_ID COMPUTATION DO NOT MATCH, DIFFERENCE IS %d" % (abs(curr_sentence_id - len(text.split(". "))))) 
    word_df["sentence_id"] = sentence_id
    word_df["sentence_length"] = sentence_length
    
    ## ACOUSTIC FEATURE AUGMENTED TRANSCRIPTION
    af_augmented_transcription = ""
    text_transcription = ""
    max_sentence_id = word_df["sentence_id"].iloc[-1]
    for sid in range(max_sentence_id + 1):
        pitch_flagged = False
        previous_pitch = ""
        pace_flagged = False
        previous_pace = ""
        
        dfr = word_df[word_df["sentence_id"] == sid]
        start = int(dfr["start"].iloc[0])
        en = int(dfr["end"].iloc[-1])
        curr_time_str = "<" + str(start) + " - " + str(en) + " seconds> "
        af_augmented_transcription_curr = "<SID " + str(sid) + ">: " + curr_time_str
        text_transcription_curr = "<SID " + str(sid) + ">: " + curr_time_str
        for idx in range(len(dfr)):
            if dfr["PITCH"].iloc[idx] is not None:
                if not pitch_flagged:
                    af_augmented_transcription_curr += "<BEGIN_" + dfr["PITCH"].iloc[idx]
                    pitch_flagged = True
                    previous_pitch = dfr["PITCH"].iloc[idx].split(":")[0]
                elif dfr["PITCH"].iloc[idx].split(":")[0] != previous_pitch:
                    af_augmented_transcription_curr += "<END_" + dfr["PITCH"].iloc[idx - 1] + " <BEGIN_" + dfr["PITCH"].iloc[idx]
                    previous_pitch = dfr["PITCH"].iloc[idx].split(":")[0]       
            else:
                if pitch_flagged:
                    af_augmented_transcription_curr += "<END_" + dfr["PITCH"].iloc[idx - 1]
                    pitch_flagged = False
                    
            if dfr["SYLL_RATE"].iloc[idx] is not None:
                if not pace_flagged:
                    af_augmented_transcription_curr += "<BEGIN_" + dfr["SYLL_RATE"].iloc[idx]
                    pace_flagged = True
                    previous_pace = dfr["SYLL_RATE"].iloc[idx].split(":")[0]
            else:
                if pace_flagged:
                    af_augmented_transcription_curr += "<END_" + dfr["SYLL_RATE"].iloc[idx - 1]
                    pace_flagged = False
                    
            af_augmented_transcription_curr += dfr["word"].iloc[idx] + dfr["PAUSE"].iloc[idx] 
            text_transcription_curr += dfr["word"].iloc[idx] + " "
        if pitch_flagged:
            af_augmented_transcription_curr += "<END_" + dfr["PITCH"].iloc[idx]
        if pace_flagged:
            af_augmented_transcription_curr += "<END_" + dfr["SYLL_RATE"].iloc[idx]
        af_augmented_transcription += af_augmented_transcription_curr + "\n"
        text_transcription += text_transcription_curr + "\n"
    af_augmented_transcription = af_augmented_transcription.strip()
    text_transcription = text_transcription.strip()            
    print(af_augmented_transcription)
    print(text_transcription)
    


    '''
    af_augmented_transcription = ""
    pitch_flagged = False
    previous_pitch = ""
    for idx in range(len(word_df)):
        if word_df["PITCH"].iloc[idx] is not None:
            if not pitch_flagged:
                af_augmented_transcription += "<BEGIN_" + word_df["PITCH"].iloc[idx]
                pitch_flagged = True
                previous_pitch = word_df["PITCH"].iloc[idx].split(":")[0]
            elif word_df["PITCH"].iloc[idx].split(":")[0] != previous_pitch:
                af_augmented_transcription += "<END_" + word_df["PITCH"].iloc[idx - 1] + " <BEGIN_" + word_df["PITCH"].iloc[idx]
                previous_pitch = word_df["PITCH"].iloc[idx].split(":")[0]       
        else:
            if pitch_flagged:
                af_augmented_transcription += "<END_" + word_df["PITCH"].iloc[idx - 1]
                pitch_flagged = False
                
        af_augmented_transcription += word_df["word"].iloc[idx] + word_df["PAUSE"].iloc[idx]         
    print(af_augmented_transcription)
    '''

    # ##### MAY BE REDUNDANT ###
    # word_df["word_length"] = word_df["word"].apply(lambda x: len(x))
    # word_df["duration"] = word_df.apply(lambda row: max(row["end"] - row["start"], 1e-5), axis = 1)
    # word_df["syllables_per_sec"] = word_df.apply(lambda row: row["num_syllables"]/row["duration"], axis = 1)
    # word_df["pitch"] = word_df.apply(lambda row: compute_word_level_audio_feat(row["start"], row["end"], pitch, feat_type = "pitch"), axis = 1)
    # word_df["energy"] = word_df.apply(lambda row: compute_word_level_audio_feat(row["start"], row["end"], energy, feat_type = "energy", silence_threshold = silence_thresh), axis = 1)
    # word_df["silence_in_word"] = word_df.apply(lambda row: compute_word_level_audio_feat(row["start"], row["end"], silence, feat_type = "silence", window_interval = window_interval), axis = 1)
    # #make and save sentence wise plot
    # print()
    
    # #print(max(word_df["pitch"]), min(word_df["pitch"][word_df["pitch"] != 0]), max(word_df["PHRASE_PITCH"]), min(word_df["PHRASE_PITCH"][word_df["PHRASE_PITCH"] != 0]))
    # # print_global_pitch_stats(word_df)
    # # print_global_pause_stats(word_df)
    # # print_global_pace(word_df)
    # # print_global_audible_stats(word_df, speechthresh)
    # word_df.to_csv(output_file_word_level_feat, index = False)
    # print("************************ WORD LEVEL FEATURES SAVED IN " + output_file_word_level_feat + " **********************************") 
    with open(output_file_aat, "w") as f:
        f.write(af_augmented_transcription)
    print("************************ AUGMENTED TRANSCRIPTION SAVED IN " + output_file_aat + " **********************************")
    with open(output_file_tt, "w") as f:
        f.write(text_transcription)
    print("************************ SENTENCE WISE TEXT TRANSCRIPTION SAVED IN " + output_file_tt + " **********************************")
    return af_augmented_transcription
    
if __name__ == "__main__":
    #print(modify_sentence("Hello!World. This-is a test. How are you? Quantus50 is the 1st in rank and 2 nd in price."))
    #print(modify_sentence("I finished 1st in the race. She came 2nd. I have 3 apples and 21 oranges. Today is my 4th visit."))
    
    #for word in ["condus", "Quantus", "triceratops", "bird"]:
    #    print(syllable_count(word))
    #print(find_boundaries([1,1,1,0,0,1,0,1,1]))
    #print(find_boundaries([0,0,0,1,1,1,1,1,0,0,1]))
    '''
    audio = "bad"
    pitch_fl = "artifacts/audio_feats/pitch_" + audio + ".txt"
    energy_fl = "artifacts/audio_feats/energy_" + audio + ".txt"
    silence_fl = "artifacts/audio_feats/silence_" + audio + ".txt"
    transcription_fl ="artifacts/transcriptions/transcription_" + audio + ".json"
    op_file = "artifacts/audio_feats/word_level_feats_" + audio + ".csv"
    word_feat_df, aat = word_level_feat_computation(transcription_fl, pitch_fl, energy_fl, silence_fl, op_file)
    '''
    pass    

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

from collections import Counter
from num2words import num2words
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import json, string, re
from jiwer import wer

import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('cmudict')

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.corpus import cmudict

STOP_WORDS = set(stopwords.words('english'))
CMU_DICTIONARY = cmudict.dict()

#from big_phoney import BigPhoney
#phoney = BigPhoney()


GLOBAL_LOW_PITCH_THRESH = 80
GLOBAL_HIGH_SYLL_RATE = 5
GLOBAL_LOW_SYLL_RATE = 3
GLOBAL_MAX_PAUSE = 2

def count_syllables(word):

    word = word.lower()
    if word.strip() == "":
        return 0
        
    if word in CMU_DICTIONARY:
        #print(CMU_DICTIONARY[word])
        return np.mean([len(list(y for y in x if y[-1].isdigit())) for x in CMU_DICTIONARY[word.lower()]])
        #return len([y for x in CMU_DICTIONARY[word] for y in x if y[-1].isdigit()])
          
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if len(word) > 1 and word[-1:] == "e":    #remove silent e
        count -= 1    
    if count == 0:
        count += 1
    return count
    
# Function to convert ordinals like 1st, 2nd, 3rd, 4th
def convert_ordinal(match):
    number = int(match.group(1))
    #print(number, num2words(number, ordinal=True))
    return num2words(number, ordinal=True)

# Function to convert plain integers like 3, 21, etc.
def convert_cardinal(match):
    number = int(match.group())
    return num2words(number)

def modify_sentence(sentence):
    #print(sentence)
    #sentence = re.sub(r'\s*%\b', ' percentage', sentence) 
    
    # Replace ordinals first (e.g., 1st, 2nd, 3-rd, 4 th, etc.)
    sentence = re.sub(r'\b(\d+)[-\s]?(st|nd|rd|th)(?=\W|$)', convert_ordinal, sentence)
    #print(sentence)
    
    '''
    # Add space between letter-number and number-letter combos -- "a123b" to "a 123 b"
    sentence = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', sentence)  # letter followed by digit
    sentence = re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', sentence)  # digit followed by letter
    '''
    
    # Replace plain numbers next (e.g., 3, 21)
    sentence = re.sub(r'\b\d+(?=\W|$)', convert_cardinal, sentence)
    
    # Define a pattern to match punctuation followed by a character or digit.
    pattern = r'([.,!?;:\-])(?=\S)'  # Match punctuation followed by non-whitespace

    # Replace the matched pattern with a space.
    modified_sentence = re.sub(pattern, r' ', sentence)
    
    # Return the modified sentence.
    return modified_sentence
    
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

def read_feat(ffile, is_pitch):
    with open(ffile) as f:
        feat = [float(x) for x in f.readlines()]
               
    if is_pitch:
        ## TODO: Get more acurate pitch using voicing
        pitch_thresh = np.percentile([p for p in feat if p != 0], 95)
        feat = np.array([min(p, pitch_thresh) for p in feat]) 
        
    return feat
    
def find_boundaries(arr):
    """
    Finds the boundaries where 1 starts and ends in a NumPy array of 0s and 1s.

    Args:
        arr: A 1D NumPy array of 0s and 1s.

    Returns:
        A list of tuples, where each tuple contains the starting and ending indices of a sequence of 1s.
    """
    
    start_indices = np.where(np.diff(arr) == 1)[0] + 1  # Find where 0 becomes 1 (start)
    end_indices = np.where(np.diff(arr) == -1)[0] + 1 # Find where 1 becomes 0 (end)
    
    boundaries = []
    
    # Handle cases where the array starts with 1 or ends with 1
    if arr[0] == 1 and len(arr) > 0:
        start_indices = np.insert(start_indices, 0, 0)
    if arr[-1] == 1 and len(arr) > 0:
        end_indices = np.append(end_indices, len(arr))

    # Combine start and end indices
    boundaries = list(zip(start_indices, end_indices))
    return boundaries

def word_level_feat_computation(timed_transcription, pitch_file, energy_file, silence_file, output_file_word_level_feat, output_file_aat, window_interval = 0.02):
    
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
    #word_df["SYLL_RATE"] = word_df["PHRASE_SYLL_RATE"].apply(lambda x: " <HIGH_PACE: " + str(x) + " syll/s> " if x > HIGH_SYLL_RATE else " <LOW_PACE: " + str(x) + " syll/sec> " if x < LOW_SYLL_RATE else None)
    
    ## ACOUSTIC FEATURE AUGMENTED TRANSCRIPTION
    af_augmented_transcription = ""
    pitch_flagged = False
    previous_pitch = ""
    energy_flagged = False
    
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
    
    ##### MAY BE REDUNDANT ###
    word_df["word_length"] = word_df["word"].apply(lambda x: len(x))
    word_df["duration"] = word_df.apply(lambda row: max(row["end"] - row["start"], 1e-5), axis = 1)
    word_df["syllables_per_sec"] = word_df.apply(lambda row: row["num_syllables"]/row["duration"], axis = 1)
    word_df["pitch"] = word_df.apply(lambda row: compute_word_level_audio_feat(row["start"], row["end"], pitch, feat_type = "pitch"), axis = 1)
    word_df["energy"] = word_df.apply(lambda row: compute_word_level_audio_feat(row["start"], row["end"], energy, feat_type = "energy", silence_threshold = silence_thresh), axis = 1)
    word_df["silence_in_word"] = word_df.apply(lambda row: compute_word_level_audio_feat(row["start"], row["end"], silence, feat_type = "silence", window_interval = window_interval), axis = 1)
    #make and save sentence wise plot
    
    #print(max(word_df["pitch"]), min(word_df["pitch"][word_df["pitch"] != 0]), max(word_df["PHRASE_PITCH"]), min(word_df["PHRASE_PITCH"][word_df["PHRASE_PITCH"] != 0]))
    print_global_pitch_stats(word_df)
    print_global_pause_stats(word_df)
    print_global_pace(word_df)
    print_global_audible_stats(word_df, speechthresh)
    word_df.to_csv(output_file_word_level_feat, index = False)
    print("************************ WORD LEVEL FEATURES SAVED IN " + output_file_word_level_feat + " **********************************") 
    with open(output_file_aat, "w") as f:
        f.write(af_augmented_transcription)
    print("************************ AUGMENTED TRANSCRIPTION SAVED IN " + output_file_aat + " **********************************")
    return af_augmented_transcription
    
def print_sentences_for_global(main_df, flagged_df, min_words_display = 3):
    sentence_ids = list(set(flagged_df["sentence_id"]))
    print_thresh = 3 ## ONLY PRINT 3 examples
    print("*" * 40)
    print("SOME EXAMPLES")
    print("*" * 40)
    for idx, sid in enumerate(sentence_ids):
        if idx == print_thresh:
            break
        df_lpr = main_df[main_df["sentence_id"] == sid]
        wd_lpr = flagged_df[flagged_df["sentence_id"] == sid]
        if len(wd_lpr) < min_words_display:
            continue
        print("SPECIFIC FLAGGED WORDs: " + ((", ").join(wd_lpr["word"]))) 
        print("IN THE SENTENCE: " + (" ").join(df_lpr["word"]))
        print("-"*50)
    
def print_global_pitch_stats(word_df):
    print("*"*50 + " GLOBAL PICTH STATS " + "*"*50)
    
    print("MAX PITCH: %.2f Hz" % (max(word_df["PHRASE_PITCH"])))
    print("MIN PITCH: %.2f Hz" % (min(word_df["PHRASE_PITCH"][word_df["PHRASE_PITCH"] != 0])))
    
    pitch_sd = np.std(word_df["PHRASE_PITCH"][word_df["PHRASE_PITCH"] != 0])
    pitch_mean = np.mean(word_df["PHRASE_PITCH"][word_df["PHRASE_PITCH"] != 0])
    print("PITCH COEFFICIENT OF VARIATION: %.2f Hz" % (pitch_sd/pitch_mean))
    if pitch_sd/pitch_mean < 0.4:
        print("MONOTONOUS SPEECH. LIVEN UP!!")
    elif pitch_sd/pitch_mean > 1.6 :
        print("TOO LIVELY. CALM DOWN !!")
    else:
        print("GOOD JOB IN MAINTAING PITCH VARIATION")
    
    low_pitch_word_df = word_df[(word_df["PHRASE_PITCH"] != 0) & (word_df["PHRASE_PITCH"] <= GLOBAL_LOW_PITCH_THRESH)]
    if len(low_pitch_word_df) > 0:
        print("="*70)
        print("ATTENTION !! PICTH GOING TOO LOW ")
        print_sentences_for_global(word_df, low_pitch_word_df)
        print("="*70)
        
def print_global_pause_stats(word_df):
    pause_more_df = word_df[word_df["PAUSE_DURATION"] >= GLOBAL_MAX_PAUSE]
    if len(pause_more_df) > 0:
        print("="*70)
        print("ATTENTION !! PAUSE TOO LONG BETWEEN WORDS ")
        print_sentences_for_global(word_df, pause_more_df, 0)
        print("="*70)
        
def print_global_audible_stats(word_df, GLOBAL_LOW_ENERGY):
    low_audible_df = word_df[word_df["PHRASE_ENERGY"] <= GLOBAL_LOW_ENERGY]
    if len(low_audible_df) > 0:
        print("="*70)
        print("ATTENTION !! LOW VOLUME ")
        print_sentences_for_global(word_df, low_audible_df)
        print("="*70)
            
def print_global_pace(word_df):
    low_pace_df = word_df[word_df["PHRASE_SYLL_RATE"] < GLOBAL_LOW_SYLL_RATE]
    high_pace_df = word_df[word_df["PHRASE_SYLL_RATE"] > GLOBAL_HIGH_SYLL_RATE]
    if len(low_pace_df) > 0:
        percent_low_pace = len(low_pace_df)/len(word_df)
        print("="*70)
        print("You speak very slow %.2f percentage of time" % (percent_low_pace))
        print_sentences_for_global(word_df, low_pace_df)
        print("="*70)
        
    if len(high_pace_df) > 0:
        percent_high_pace = len(high_pace_df)/len(word_df)
        print("="*70)
        print("You speak very fast %.2f percentage of time" % (percent_high_pace))
        print_sentences_for_global(word_df, high_pace_df)
        print("="*70)

def remove_stopwords(text):
    tokens = nltk.word_tokenize(text.lower())
    filtered = [word for word in tokens if word.isalnum() and word not in STOP_WORDS]
    return ' '.join(filtered)
    
def sentence_exact_match(ground_truth_file, timed_transcription):
    with open(ground_truth_file) as f:
        ground_truth = f.read()
    ground_truth = modify_sentence(ground_truth)
    
    with open(timed_transcription) as f:
        transcription = json.load(f)
    transcription_text = modify_sentence(transcription["text"])
    
    ground_truth_sentences = [s.translate(str.maketrans('', '', string.punctuation)).lower() for s in sent_tokenize(ground_truth)] 
    transcription_sentences = [s.translate(str.maketrans('', '', string.punctuation)).lower() for s in sent_tokenize(transcription_text)]
    transcription_sentences_no_stopwords = [remove_stopwords(s.translate(str.maketrans('', '', string.punctuation)).lower()) for s in sent_tokenize(transcription_text)]
    
    total_wer = 0
    for idx in range(len(ground_truth_sentences)):
        gt = remove_stopwords(ground_truth_sentences[idx])
        min_wer = float("inf")
        min_idx = 0
        #print(ground_truth_sentences[idx])
        for j_idx in range(len(transcription_sentences_no_stopwords)):
            #print(transcription_sentences_no_stopwords[j_idx])
            #print(wer(gt, transcription_sentences_no_stopwords[j_idx]))
            #print("-"*50)
            if wer(gt, transcription_sentences_no_stopwords[j_idx]) < min_wer:
                min_wer = wer(gt, transcription_sentences_no_stopwords[j_idx])
                min_idx = j_idx
                
        total_wer += min_wer
        
    mean_wer = total_wer/len(ground_truth_sentences)
    WORD_ERROR_RATE_THRESH = 0.4
    WORD_ERROR_RATE_THRESH_MID = 0.7
    
    if mean_wer < WORD_ERROR_RATE_THRESH:
        print("GOOD JOB. VERY CLOSE TO GROUND TRUTH")
    elif mean_wer < WORD_ERROR_RATE_THRESH_MID:
        print("DECENT JOB BUT CAN BE BETTER")
        # Call LLM
    else:
        print("MAN YOU SUCK")
        #Call LLM
    print("WORD ERROR RATE %.2f" % (total_wer / len(ground_truth_sentences)))
    
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
    sentence_exact_match("artifacts/transcriptions/ground_truth_1.txt", "artifacts/transcriptions/transcription_bad.json")

    

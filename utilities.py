import numpy as np
import nltk
import os, re, json, string
import pandas as pd
from num2words import num2words
nltk.download('punkt_tab')
nltk.download('stopwords')
#nltk.download('cmudict')
from jiwer import wer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.corpus import cmudict

STOP_WORDS = set(stopwords.words('english'))
CMU_DICTIONARY = cmudict.dict()
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
    
    sentence = sentence.replace("%","")
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
    
def longest_zero_sequence(arr):
    max_zeros = 0
    current_zeros = 0
    
    for num in arr:
        if num == 0:
            current_zeros += 1
            max_zeros = max(max_zeros, current_zeros)
        else:
            current_zeros = 0
    
    return max_zeros
    
def read_feat(ffile, is_pitch):
    with open(ffile) as f:
        feat = [float(x) for x in f.readlines()]
               
    if is_pitch:
        ## TODO: Get more acurate pitch using voicing
        pitch_thresh = np.percentile([p for p in feat if p != 0], 95)
        feat = np.array([min(p, pitch_thresh) for p in feat]) 
        
    return feat
    
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
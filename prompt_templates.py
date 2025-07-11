AUDIO_PROMPT = '''I want you to act like a voice analyzer expert. You know that voice modulation is all about how you use your voice to convey emotion, emphasize key points, and keep your listener engaged. You know what words people should raise or lower their pitch, where they should pause for an effective sales pitch. 
I will give you a transcription augmented with acoustic features. If the speaker pauses after a word, it will be marked as <PAUSE: [duration]ms>. If the person's pitch goes too high or low compared to their baseline pitch, the word boundaries will be marked with <BEGIN_HIGH/LOW_PITCH: [f]Hz> and <END_HIGH/LOW_PITCH: [f]Hz>. Now given the augmented transcription tell me your feedback. What are the positive things and negative things that could be improved? 

Augmented Transcription: {augmented_transcription}
'''

TEXT_PROMPT = '''You are a pitch sales evaluator. I wil provide you with reference sales pitch and the sales pitch made by a sales rep based on that. You need to evaluate the sales rep's pitch and provide your feedback along three dimension.\n\n1. Correctness: Based on reference text, list out all the factaully incorrect things the sales rep said. It should also include things that the rep said but it is not grounded in any evidence as provided by the reference sales pitch.\n2. Completeness: Based on reference text, list out all the contents in reference text that the rep has missed. Note that minor grammatical or niceties should not be flagged. Only flag if an important ot relevant sentence or idea from the reference pitch was not conveyed by the sales rep.\n3. Information flow: Based on reference text, are the ideas conveyed in the same order? While making the pitch did the rep convey information in a way that makes their message difficult to come across?\n\n<REFERENCE SALES PITCH>: {ground_truth_text}\n\n<SALES PITCH MADE BY REP>: {asr_transcription}'''

TEXT_COMPLETENESS_PROMPT = '''You are a pitch sales evaluator. I wil provide you with reference text and the pitch sales made by a sales rep based on that. \n\n<REFERENCE SALES PITCH>: {ground_truth_text}\n\n<SALES PITCH MADE BY REP>: {asr_transcription}'''

TEXT_INFORMATION_FLOW = '''You are a pitch sales evaluator. I wil provide you with reference sales pitch and the sales pitch made by a sales rep based on that. Based on reference text, list out all the factaully incorrect things the sales rep said. It should also include things that the rep said but it is not grounded in any evidence as provided by the reference sales pitch.\n\n<REFERENCE SALES PITCH>: {ground_truth_text}\n\n<SALES PITCH MADE BY REP>: {asr_transcription}'''

TEXT_QUALITY_PROMPT = '''Given a sales pitch transcription, your job is to flag any disfluencies and mispronunciations such as the following:\n\n1. Any filler words appearing in transcription\n2. Any revision i.e. when a person says a phrase but changes their sentence e.g. "And we were I was fortunate..." or "If you how long do you want to stay?"\n3. Any repition e.g. "well with my with my grandmother..."\n4. Any fumbling such as aplogizing something they should have said\n5. Looking at the reference text and the actual sales, any mispronunciation they have made\n6. Any other disfluency that you notice.\n\nREFERENCE SALES PITCH>: {ground_truth_text}\n\n<ACTUAL SALES TRANSCRIPTION>: {asr_transcription}'''

# correctness
# completeness
# information flow

# https://arxiv.org/pdf/2311.00867
# mispronunciation, false start, revision, repetition

## At most 3 instances of good and bad
## json format {"feedback", "how to improve", "sentence in question"}

## 
AUDIO_PROMPT_V1 = '''Act as a voice modulation expert analyzing sales pitches. You understand how vocal delivery—pitch, pauses, and emphasis—impacts persuasion and listener engagement. I will give you an augmented transcription with acoustic features: (i) Pauses are marked as <PAUSE: [duration]ms>. (ii) High or low pitch shifts are marked using <BEGIN_HIGH_PITCH: [f]Hz>, <END_HIGH_PITCH: [f]Hz>, <BEGIN_LOW_PITCH: [f]Hz>, and <END_LOW_PITCH: [f]Hz>.\nBased on this, provide feedback on the speaker’s vocal delivery. Focus on how effectively they used pitch modulation and pauses to create emphasis and emotional impact.\nOutput must be a Python list of dictionaries. Each dictionary must have the following fields:\n"feedback": A concise observation.\n"helpful tip": A practical, clear suggestion.\n"evidence": the sentence from transcription that supports the feedback.\nGive at most 3 examples each of good and bad vocal delivery (fewer is fine if less feedback is needed). Keep all feedback focused and non-redundant. Only output the feedback in the json format for the positive and negative aspects separately and nothing else. Here is the augmented transcription:\n\n
{augmented_transcription}
'''

AUDIO_PROMPT_V2 = """Act as a voice modulation expert analyzing sales pitches. You understand how vocal delivery—pitch, pauses, and emphasis—impacts persuasion and listener engagement. I will give you an augmented transcription with acoustic features: 
(i) Pauses are marked as <PAUSE: [duration]ms>. 
(ii) High or low pitch shifts are marked using <BEGIN_HIGH_PITCH: [f]Hz>, <END_HIGH_PITCH: [f]Hz>, <BEGIN_LOW_PITCH: [f]Hz>, and <END_LOW_PITCH: [f]Hz>.
(iii) High or low pace regions of speech are marked with syllables per second: <BEGIN_HIGH_PACE: [p] syl/s>, <END_HIGH_PACE: [p] syl/s>, <BEGIN_LOW_PACE: [p] syl/s>, <END_LOW_PACE: [p] syl/s>.

Below there are some instructions provided on good speaking or sales pitching techniques that can be used to provide feedback.
(a) For adding energy and drawing attention to specific words or phrases, a higher pitch can be used.
(b) To convey seriousness, authority, or to underscore a weighty point, a lower pitch is often effective.
(c) Ideal pacing should lie between 3-6 syllables per second. It is also fine if speaker is reducing pace to abput 2 syllables per second to emphasize on key words.
(d) Ideal pause between sentences should last 1-2 seconds. Within a sentence, pause should be about 0.5 second. It is alright if speaker pauses a bit more to emphasize on keywords.
(e) Note that syllables per second can be affected by pauses. Do not penalise if syllables per second is a bit lower on account of pausing to put emphasis on keywords. However, under no circumstances pause should be greater than 3 seconds.

Based on this, provide feedback on the speaker’s vocal delivery. Focus on how effectively they used pitch modulation, pacing and pauses to create emphasis and emotional impact. Focus especially on the regions containing keywords or emotional words. Keep all feedback concise, focused and non-redundant. While providing negative feedback do not be overly critical. Be motivating in your feedback. Do not hallucinate and make sure everything is grounded in evidence provided in augmented transcript. Give at most 3 examples each of good and bad vocal delivery (fewer is fine if less feedback is needed). Output must be a Python list of dictionaries in the following format (except spaces and tabs). All the fields are mandatory to include unless noted otherwise.
{{"positive_feedback": [
    {{"id": <int>, // sentence ID or <SID> of the sentence about which feedback is provided as it appears in the augmented transcript,
      "phrase": <str>, // The phrase in the sentence (or the sentence) based on which feedback is provided. Include pitch, pace and pause information as it appears in augmented transcript
      "feedback": <str>, // A concise observation about what was done right. DO NOT include any numerical pitch, pace and pause information from the augmented transcript in the feedback. Instead use words like stress, pacing, intonation, pause, emotiona conveyed etc.
      "score": <int>, // A score between 1-10 based on how important is the phrase and feedback to help the speaker improve or motivate them to do better.
      "attribute": <str> // What attribute the feedback is focussing on such as pace, modulation, pause etc.
    }},
    ... // 1 or 2 more such positive feedbacks in above format
    ],
  "negative_feedback": [
    {{"id": <int>, // sentence ID or <SID> of the sentence about which feedback is provided as it appears in the augmented transcript,
      "phrase": <str>, // The phrase in the sentence (or the sentence) based on which feedback is provided. Include pitch, pace and pause information as it appears in augmented transcript
      "feedback": <str>, // A concise observation about what was done wrong. DO NOT include any numerical pitch, pace and pause information from the augmented transcript in the feedback. Instead use words like stress, pacing, intonation, pause, emotiona conveyed etc.
      "tip": <str>, // A practical, clear suggestion on how to further improve. This is MANDATORY for negative feedback.
      "score": <int>, // A score between 1-10 based on how important is the phrase and feedback to help the speaker improve or motivate them to do better.
      "attribute": <str> // What attribute the feedback is focussing on such as pace, modulation, pause etc.
    }}, 
    ... // 1 or 2 more such negative feedbacks in above format
    ]
}}

Only output the feedback in the json format for the positive and negative aspects in above format and nothing else. Here is the augmented transcription:

{augmented_transcription}
"""

TEXT_PROMPT_V1 = """You are a pitch sales evaluator. I wil provide you with reference sales pitch and the actual pitch made by a sales rep. You need to evaluate the sales rep's pitch with respect to the reference sales pitch. Provide your feedback along three dimension.

1. Correctness: Based on reference text, list out all the factaully incorrect things the sales rep said. It should also include things that the rep said but it is not grounded in any evidence as provided by the reference sales pitch.
2. Completeness: Based on reference text, list out all the contents in reference text that the rep has missed. Note that minor grammatical or niceties should not be flagged. Only flag if an important or relevant sentence or idea from the reference pitch containing keywords was not conveyed by the sales rep.
3. Information flow: Based on reference text, are the ideas conveyed in the same order? While making the pitch did the rep convey information in a way that makes their message difficult to come across?

Based on this, provide feedback on the sales rep pitch. Focus especially on the regions containing keywords and key ideas. Keep all feedback concise, focused and non-redundant. While providing negative feedback do not be overly critical. Be motivating in your feedback. Do not hallucinate and make sure everything is grounded in evidence provided in augmented transcript. Give at most 3 examples each of good and bad aspects of the rep sales pitch (fewer is fine if less feedback is needed). Output must be a Python list of dictionaries in the following format (except spaces and tabs). All the fields are mandatory to include unless noted otherwise.
{{"positive_feedback": [
    {{"id": <int>, // sentence ID or <SID> of the sentence about which feedback is provided as it appears in the augmented transcript,
      "phrase": <str>, // The phrase in the sentence (or the sentence) based on which feedback is provided. Include pitch, pace and pause information as it appears in augmented transcript
      "feedback": <str>, // A concise observation about what was done right. DO NOT include any numerical pitch, pace and pause information from the augmented transcript in the feedback. Instead use words like stress, pacing, intonation, pause, emotiona conveyed etc.
      "score": <int>, // A score between 1-10 based on how important is the phrase and feedback to help the speaker improve or motivate them to do better.
      "attribute": <str> // What attribute the feedback is focussing on such as correctness, completeness, flow of information etc.
    }},
    ... // 1 or 2 more such positive feedbacks in above format
    ],
  "negative_feedback": [
    {{"id": <int>, // sentence ID or <SID> of the sentence about which feedback is provided as it appears in the augmented transcript,
      "phrase": <str>, // The phrase in the sentence (or the sentence) based on which feedback is provided. Include pitch, pace and pause information as it appears in augmented transcript
      "feedback": <str>, // A concise observation about what was done wrong. DO NOT include any numerical pitch, pace and pause information from the augmented transcript in the feedback. Instead use words like stress, pacing, intonation, pause, emotiona conveyed etc.
      "tip": <str>, // A practical, clear suggestion on how to further improve. This is MANDATORY for negative feedback.
      "score": <int>, // A score between 1-10 based on how important is the phrase and feedback to help the speaker improve or motivate them to do better.
      "attribute": <str> // What attribute the feedback is focussing on such as correctness, completeness, flow of information etc.
    }}, 
    ... // 1 or 2 more such negative feedbacks in above format
    ]
}}

Only output the feedback in the json format for the positive and negative aspects in above format and nothing else. 

<REFERENCE SALES PITCH> 
{ground_truth_text}

<SALES PITCH MADE BY REP>
{asr_transcription}
"""

DISFLUENCY_PROMPT = """You are a pitch sales evaluator. I wil provide you with reference sales pitch and the actual pitch made by a sales rep. Given a sales pitch transcription, your job is to flag any disfluencies and mispronunciations with respect to the reference sales pitch. DO NOT include any feedback about filler words such as uh, uhh, uhm, aah, umm. The criteria to identify disfluencies are mentioned below:

1. Any revision i.e. when a person says a phrase but changes their sentence e.g. "And we were I was fortunate..." or "If you how long do you want to stay?" or "this drug I am sorry Paracetamol is..."
2. Any repition e.g. "well with my with my grandmother..."
3. Any fumbling such as stammering, difficulty enuniciating a word 
4. Looking at the reference text and the actual sales, any mispronunciation they have made
5. Any other disfluency that you notice. Note that you should not include any feedback about filler words such uh, umm, uhm, ah etc.

Based on this, provide feedback on the sales rep pitch. Focus especially on the regions containing keywords and key ideas. Keep all feedback concise, focused and non-redundant. While providing negative feedback do not be overly critical. Be motivating in your feedback. Do not hallucinate and make sure everything is grounded in evidence provided in augmented transcript. Give at most 3 examples where such disfluencies occur (fewer is fine if less feedback is needed). Output must be a Python list of dictionaries in the following format (except spaces and tabs). All the fields are mandatory to include unless noted otherwise.
{{"quality_feedback": [
    {{"id": <int>, // sentence ID or <SID> of the sentence about which feedback is provided as it appears in the augmented transcript,
      "phrase": <str>, // The phrase in the sentence (or the sentence) based on which feedback is provided. Include pitch, pace and pause information as it appears in augmented transcript
      "feedback": <str>, // A concise observation about what was done wrong. DO NOT include any numerical pitch, pace and pause information from the augmented transcript in the feedback. Instead use words like stress, pacing, intonation, pause, emotiona conveyed etc.
      "tip": <str>, // A practical, clear suggestion on how to further improve. This is MANDATORY for negative feedback.
      "score": <int>, // A score between 1-10 based on how important is the phrase and feedback to help the speaker improve or motivate them to do better.
      "attribute": <str> // What attribute the feedback is focussing on such as repetition, false start etc.
    }}, 
    ... // 1 or 2 more such quality feedbacks in above format
    ]
}}

Only output the feedback in the json format for the positive and negative aspects in above format and nothing else. 

REFERENCE SALES PITCH>: 
{ground_truth_text}

<ACTUAL SALES TRANSCRIPTION>: 
{asr_transcription}"""


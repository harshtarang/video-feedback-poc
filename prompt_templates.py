AUDIO_PROMPT = '''I want you to act like a voice analyzer expert. You know that voice modulation is all about how you use your voice to convey emotion, emphasize key points, and keep your listener engaged. You know what words people should raise or lower their pitch, where they should pause for an effective sales pitch. 
I will give you a transcription augmented with acoustic features. If the speaker pauses after a word, it will be marked as <PAUSE: [duration]ms>. If the person's pitch goes too high or low compared to their baseline pitch, the word boundaries will be marked with <BEGIN_HIGH/LOW_PITCH: [f]Hz> and <END_HIGH/LOW_PITCH: [f]Hz>. Now given the augmented transcription tell me your feedback. What are the positive things and negative things that could be improved? 

Augmented Transcription: {augmented_transcription}
'''

TEXT_PROMPT = '''You are a pitch sales evaluator. I wil provide you with reference sales pitch and the sales pitch made by a sales rep based on that. You need to evaluate the sales rep's pitch and provide your feedback along three dimension.\n\n1. Correctness: Based on reference text, list out all the factaully incorrect things the sales rep said. It should also include things that the rep said but it is not grounded in any evidence as provided by the reference sales pitch.\n2. Completeness: Based on reference text, list out all the contents in reference text that the rep has missed. Note that minor grammatical or niceties should not be flagged. Only flag if an important ot relevant sentence or idea from the reference pitch was not conveyed by the sales rep.\n3. Information flow: Based on reference text, are the ideas conveyed in the same order? While making the pitch did the rep convey information in a way that makes their message difficult to come across?\n\n<REFERENCE SALES PITCH>: {ground_truth_text}\n\n<SALES PITCH MADE BY REP>: {asr_transcription}'''

# TEXT_COMPLETENESS_PROMPT = '''You are a pitch sales evaluator. I wil provide you with reference text and the pitch sales made by a sales rep based on that. \n\n<REFERENCE SALES PITCH>: {ground_truth_text}\n\n<SALES PITCH MADE BY REP>: {asr_transcription}'''

# TEXT_INFORMATION_FLOW = '''You are a pitch sales evaluator. I wil provide you with reference sales pitch and the sales pitch made by a sales rep based on that. Based on reference text, list out all the factaully incorrect things the sales rep said. It should also include things that the rep said but it is not grounded in any evidence as provided by the reference sales pitch.\n\n<REFERENCE SALES PITCH>: {ground_truth_text}\n\n<SALES PITCH MADE BY REP>: {asr_transcription}'''

TEXT_QUALITY_PROMPT = '''Given a sales pitch transcription, your job is to flag any disfluencies and mispronunciations such as the following:\n\n1. Any filler words appearing in transcription\n2. Any revision i.e. when a person says a phrase but changes their sentence e.g. "And we were I was fortunate..." or "If you how long do you want to stay?"\n3. Any repition e.g. "well with my with my grandmother..."\n4. Any fumbling such as aplogizing something they should have said\n5. Looking at the reference text and the actual sales, any mispronunciation they have made\n6. Any other disfluency that you notice.\n\nREFERENCE SALES PITCH>: {ground_truth_text}\n\n<ACTUAL SALES TRANSCRIPTION>: {asr_transcription}'''

# correctness
# completeness
# information flow

# https://arxiv.org/pdf/2311.00867
# mispronunciation, false start, revision, repetition

import base64
import json
import tempfile
from time import sleep
import uuid
import streamlit as st
import os
import random

from feature_processor import word_level_feat_computation
from llm_helper import prompt_for_audio, prompt_for_quality, prompt_for_text, speech_to_text
from non_llm_feedback import filler_detection
from post_processing import collate_all_feedback
from speech_helper import get_speech_features, convert_vid_to_audio
from dotenv import load_dotenv
from prompt_templates import AUDIO_PROMPT, AUDIO_PROMPT_V2, DISFLUENCY_PROMPT, TEXT_PROMPT, TEXT_PROMPT_V1, TEXT_QUALITY_PROMPT


page_bg_img = '''
<style>
.stApp {
  background-image: url("\\app\\static\\bg-5.png");
  background-size: cover;
  background-position: center;
  background-color: rgba(0, 0, 0, 0);
}
</style>
'''
# page_bg_img = ""

def get_keyword_list():
    # keyword_list = "Quantus 50, Co-enzyme Q10, Selenium, umm, hmm, Udiliv, diabetes, obesity, non-alcoholic liver diseases, liver disease, non-alcoholic fatty liver disease, AST, ALT, GGT, ALP, Ursodeoxycholic acid, position paper endorsed by 4 esteemed societies, Indian society of Gastroenterology, Indian college of cardiology, Endocrine society of India, INASL, cholestasis, hepatoprotective, antioxidant, anti-inflammatory, antiapoptotic, hypercholeretic, Non-alcoholic Liver Disease, 300mg BID, 10-15mg per , kg per day"
    keyword_list = "Boniliv,  Ursodeoxycholic acid, Indian society of Gastroenterology, the Indian college of cardiology, Endocrine Society of India, INASL, AST, ALT, GGT, ALP, 300mg BID, Non-alcoholic Liver Disease, umm, hmm, uhmm, mmmm, mhmm, uh, uhh, uhm"
    return keyword_list

def load_environment():
    """Load environment variables from .env file"""
    # Load .env file from current directory
    load_dotenv(override=True)
    print("Environment variables loaded successfully!")

js = '''
<script>
    var body = window.parent.document.querySelector("#feedback-section");
    console.log(body);
    body.scrollTop = 100;
</script>
'''

# Enhanced CSS for bubble flow UI
bubble_flow_css = '''
<style>
.instruction-container {
    padding: 20px;
    margin: 10px 0;
}

.step-bubble {
    # background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 25px;
    padding: 20px 25px;
    margin: 15px 0;
    color: white;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    position: relative;
    transition: all 0.3s ease;
    text-align: left;
}

.step-bubble:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
}

.step-bubble.step-1 {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    box-shadow: 0 8px 32px rgba(255, 107, 107, 0.3);
}

.step-bubble.step-2 {
    background: linear-gradient(135deg, #4ecdc4 0%, #2980b9 100%);
    box-shadow: 0 8px 32px rgba(78, 205, 196, 0.3);
}

.step-bubble.step-3 {
    background: linear-gradient(135deg, white 0%, white 100%);
    box-shadow: 0 8px 32px rgba(252, 182, 159, 0.3);
    color: #333;
}

.step-number {
    display: inline-block;
    width: 30px;
    height: 30px;
    background: rgba(255, 255, 255, 0.3);
    border-radius: 50%;
    text-align: center;
    line-height: 30px;
    font-weight: bold;
    margin-right: 15px;
    font-size: 48px;
    margin-top: 20px;
}

.step-header {
    font-size: 20px;
    font-weight: bold;
}

.step-content {
    display: inline-block;
    vertical-align: top;
    width: calc(100% - 50px);
    font-size: 16px;
    line-height: 1.5;
}

.arrow-down {
    position: relative;
    left: 50%;
    transform: translateX(-50%);
    width: 0;
    height: 0;
    border-left: 15px solid transparent;
    border-right: 15px solid transparent;
    border-top: 20px solid #667eea;
    margin: 10px 0;
}

.arrow-down.arrow-1 {
    border-top-color: #ff6b6b;
    animation-delay: 0.3s;
}

.arrow-down.arrow-2 {
    border-top-color: #4ecdc4;
    animation-delay: 0.8s;
}

.title-bubble {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    padding: 15px 25px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

@keyframes bounce {
    0%, 20%, 50%, 80%, 100% { transform: translateX(-50%) translateY(0); }
    40% { transform: translateX(-50%) translateY(-10px); }
    60% { transform: translateX(-50%) translateY(-5px); }
}

.glow-effect {
    position: relative;
    overflow: hidden;
}

.glow-effect::before {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #667eea, #ffecd2);
    border-radius: 27px;
    z-index: -1;
    opacity: 0.7;
}

@keyframes glow-rotate {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>

<style>
/* Custom styles for Streamlit components */
.stFileUploader > div {
    background: linear-gradient(135deg, white 0%, white 100%);
    border-radius: 25px;
    padding: 20px 25px;
    margin: 15px 0;
    color: #333;
    box-shadow: 0 8px 32px rgba(252, 182, 159, 0.3);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    transition: all 0.3s ease;
}

.stFileUploader > section {
    background: linear-gradient(135deg, white 0%, white 100%);
    border-radius: 25px;
    padding: 20px 25px;
    margin: 15px 0;
    color: #333;
    box-shadow: 0 8px 32px rgba(252, 182, 159, 0.3);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    transition: all 0.3s ease;
}

.stFileUploader > div:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(252, 182, 159, 0.4);
}

.stFileUploader > label > div {
    font-size: 22px; /* Increased font size */
    font-weight: bold;
    color: #333; /* Adjust color for better contrast */
}

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 25px;
    padding: 15px 30px;
    color: white;
    font-size: 18px;
    font-weight: bold;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    transition: all 0.3s ease;
    cursor: pointer;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
}
</style>
'''

def render_instruction_flow():
    """Render the enhanced instruction flow with bubbles and arrows"""
    st.markdown(bubble_flow_css, unsafe_allow_html=True)
    
    instruction_html = '''
    <div class="instruction-container">
        <div class="title-bubble">
            <h3 style="margin: 0; font-size: 24px; text-align: left;"> How to Use This Tool</h3>
        </div>
        <div class="step-bubble step-3">
            <div class="step-number">1</div>
            <div class="step-content">
                <div class="step-header"><strong>📹 Upload Media File</strong></div>
                Upload the detailing video or audio file that you want to analyze for feedback.
            </div>
        </div>
        <div class="arrow-down arrow-1"></div>
        <div class="step-bubble step-3">
            <div class="step-number">2</div>
            <div class="step-content">
                <div class="step-header"><strong>📄 Upload Detailing Script</strong></div>
                Upload the ideal detailing script provided by the brand marketing team for this sales call.
            </div>
        </div>
        <div class="arrow-down arrow-1"></div>
        <div class="step-bubble step-3">
            <div class="step-number">3</div>
            <div class="step-content">
                <div class="step-header"><strong>🚀 Generate Feedback</strong></div>
                Click the 'Generate Feedback' button to receive feedback on the uploaded detailing video/audio.
            </div>
        </div>
    </div>
    '''
    
    st.markdown(instruction_html, unsafe_allow_html=True)

def render_upload_file():
    """Render the enhanced instruction flow with bubbles and arrows"""
    st.markdown(bubble_flow_css, unsafe_allow_html=True)
    
    instruction_html = '''
    <div class="instruction-container">
        <div class="title-bubble">
            <h3 style="margin: 0; font-size: 24px; text-align: left;"> Upload Files</h3>
        </div>
        
    </div>
    '''
    
    st.markdown(instruction_html, unsafe_allow_html=True)

def main():
    
    st.set_page_config(layout="wide")
    st.markdown(page_bg_img, unsafe_allow_html=True)
    # Initialize session state for files and feedback
    if 'uploaded_media' not in st.session_state:
        st.session_state.uploaded_media = None
    if 'uploaded_text' not in st.session_state:
        st.session_state.uploaded_text = None
    if 'feedback' not in st.session_state:
        st.session_state.feedback = None

    # UI Layout
    col1, col2 = st.columns([1, 2])

    with col1:
        render_instruction_flow()
        
        # Show Settings only in developer mode
        if os.environ.get('DEVELOPER_MODE') == 'True':
            st.header("Settings")
            model_option = st.selectbox(
                "Select LLM Model",
                ["OpenAI", "Gemini"],
                index=0  # default to OpenAI
            )
            
            st.subheader("Prompt Templates")
            audio_prompt = st.text_area("Speech Analyzer Prompt", value=AUDIO_PROMPT_V2, height=200)
            text_prompt = st.text_area("Pitch Analyzer Prompt", value=TEXT_PROMPT_V1, height=200)
            quality_prompt = st.text_area("Quality Analyzer Prompt", value=DISFLUENCY_PROMPT, height=200)
        else:
            model_option = "OpenAI"
            audio_prompt = AUDIO_PROMPT_V2
            text_prompt = TEXT_PROMPT_V1
            quality_prompt = DISFLUENCY_PROMPT
    with col2:

        # st.header("Upload Files")
        render_upload_file()

        # File Uploaders
        st.markdown('<div class="stFileUploader">', unsafe_allow_html=True)
        uploaded_media = st.file_uploader(
            "Upload Detailing Video/Audio File", type=["mp4", "mp3", "wav", "avi", "mov"]
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="stFileUploader">', unsafe_allow_html=True)
        uploaded_text = st.file_uploader("Upload Detailing Script File", type=["txt"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Only show the button if both files are uploaded
        if uploaded_media and uploaded_text:
            # Store current files in session state
            st.session_state.uploaded_media = uploaded_media
            st.session_state.uploaded_text = uploaded_text

            if st.button("Generate Feedback", type="primary"):
                with st.spinner("Generating feedback... Please wait for a few minutes and do not refresh or close the page."):
                    st.markdown('<style>div.stSpinner > div > div {font-size: 22px;}</style>', unsafe_allow_html=True) # Increased font size for spinner
                    st.session_state.feedback = ([], [])  # Reset feedback
                    if os.getenv("USE_CACHED_FEEDBACK") == "True" and os.path.exists("cache/feedback/audio.txt"):
                        print("*************************** USING CACHED FEEDBACK **********************************")
                        tt_file = "cache/feedback/timed_transcription.txt"
                        audio_feedback_file = "cache/feedback/audio.txt"
                        text_feedback_file = "cache/feedback/text.txt"
                        quality_feedback_file = "cache/feedback/quality.txt"
    
                        sleep(5)  # Simulate processing time
                            
                    else:

                        print(f"File name with ext: {uploaded_media.name}")
                        file_name = uploaded_media.name.split('.')[0]
                        print(f"File name: {file_name}")
                        # Save uploaded media to a temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_media.name)[-1]) as temp_input:
                            temp_input.write(uploaded_media.read())
                            temp_input_path = temp_input.name

                        # Save uploaded text to a temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_input:
                            temp_input.write(uploaded_text.read())
                            ground_truth_path = temp_input.name

                        # Create a temp output path for audio
                        temp_output_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        temp_output_audio_path = temp_output_audio.name
                        temp_output_audio.close()

                        # Convert video/audio to wav
                        convert_vid_to_audio(temp_input_path, temp_output_audio_path)

                        # Generate speech features
                        pitch_txt = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
                        energy_txt = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
                        silence_txt = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
                        get_speech_features(temp_output_audio_path, pitch_txt, energy_txt, silence_txt)

                        # Generate transcription text from audio
                        transcription_fl = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
                        speech_to_text(temp_output_audio_path, transcription_fl, keyword_list = get_keyword_list(), file_name=file_name)

                        # Generate word-level features
                        word_level_feat_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
                        aat_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
                        tt_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
                        audio_feedback_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
                        text_feedback_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
                        quality_feedback_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name

                        word_level_feat_computation(transcription_fl, pitch_txt, energy_txt, silence_txt, word_level_feat_file, aat_file,  tt_file)
                        # non_llm_feedback = filler_detection(word_level_feat_file, silence_txt, window_interval = 0.02)
                        # if non_llm_feedback:
                        #     st.text_area("Filler Words Feedback", value=non_llm_feedback, height=200)
                        ## LLM BASED FEEDBACK
                        prompt_for_audio(aat_file, audio_feedback_file, model=model_option, audio_prompt=audio_prompt)
                        prompt_for_text(ground_truth_path, tt_file, text_feedback_file, model=model_option, text_prompt=text_prompt)
                        prompt_for_quality(ground_truth_path, tt_file, quality_feedback_file, model=model_option, quality_prompt=quality_prompt)
                        
                    pos_feedback, neg_feedback = collate_all_feedback(tt_file, audio_feedback_file, text_feedback_file, quality_feedback_file)
                    st.session_state.feedback = (pos_feedback, neg_feedback)
                    st.markdown('<div style="background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px; font-size: 22px; font-weight: bold; opacity: 1;">Feedback generated successfully!</div>', unsafe_allow_html=True)
                    # st.success("Feedback generated successfully!") # Replaced with custom markdown for full opacity and larger text
                    # st.markdown(f"Feedback:\n{all_feedback}")

        # If we have generated feedback in session state, display it
        if st.session_state.feedback:
            
            st.markdown("""
                <style>
                .feedback-card {
                    padding: 15px;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
                    margin: 15px 0;
                    border: 1px solid #e0e0e0;
                    background-color: white;
                }
                .feedback-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 10px;
                }
                .feedback-score {
                    width: 40px;
                    height: 40px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: bold;
                    font-size: 18px;
                }
                .feedback-time {
                    font-size: 12px;
                    color: #777;
                    margin-top: 5px;
                }
                .feedback-tip {
                    background-color: #f8f9fa;
                    padding: 10px;
                    border-radius: 5px;
                    margin-top: 10px;
                    border-left: 3px solid #4CAF50;
                }
                </style>
            """, unsafe_allow_html=True)
            
            pos_feedback, neg_feedback = st.session_state.feedback
            
            if 'show_all_feedback' not in st.session_state:
                st.session_state.show_all_feedback = False

            if st.session_state.show_all_feedback:
                display_pos_feedback = pos_feedback
                display_neg_feedback = neg_feedback
            else:
                display_pos_feedback = filter_feedback(pos_feedback, "positive")
                display_neg_feedback = filter_feedback(neg_feedback, "negative")

            st.header("Feedback")
            st.subheader("🎯 Strengths: What you did well")
            show_feedback_container(display_pos_feedback)
            st.subheader("🛠️ Areas of improvement: What you can do differently")
            show_feedback_container(display_neg_feedback)
            
            if st.button("Show All Feedback"):
                st.session_state.show_all_feedback = not st.session_state.show_all_feedback
                st.rerun()

def filter_feedback(feedback_list, feedback_type):
    text_exactness_feedback = [f for f in feedback_list if 'text_completeness' in f.get('type') or 'text_correctness' in f.get('type')]
    pacing_feedback = [f for f in feedback_list if "pace" in f.get('type') or "pause" in f.get('type')]
    disfluencies_feedback = [f for f in feedback_list if "quality" in f.get('type')]
    
    filtered_feedback = []

    if feedback_type == "positive":
        # Prioritization for Strengths
        # P1: Text Exactness (randomly choose any 2)
        if len(text_exactness_feedback) >= 2:
            filtered_feedback.extend(random.sample(text_exactness_feedback, 2))
        else:
            filtered_feedback.extend(text_exactness_feedback)

        # P2: Pacing (randomly choose any 1)
        if len(pacing_feedback) >= 1:
            filtered_feedback.extend(random.sample(pacing_feedback, 1))
        elif pacing_feedback:
            filtered_feedback.extend(pacing_feedback)

        # P3: Disfluencies (randomly choose 1 or more)
        remaining_slots = 4 - len(filtered_feedback)
        if remaining_slots > 0:
            if len(disfluencies_feedback) >= remaining_slots:
                filtered_feedback.extend(random.sample(disfluencies_feedback, remaining_slots))
            else:
                filtered_feedback.extend(disfluencies_feedback)

    elif feedback_type == "negative":
        # Prioritization for Areas of Improvement
        # P1: Text Exactness (choose top 2 based on high Risk Score)
        text_exactness_feedback_sorted = sorted(text_exactness_feedback, key=lambda x: x.get('score', 0), reverse=True)
        filtered_feedback.extend(text_exactness_feedback_sorted[:2])

        # P2: Pacing (choose 1 or max 2 with the highest risk scores)
        pacing_feedback_sorted = sorted(pacing_feedback, key=lambda x: x.get('score', 0), reverse=True)
        remaining_slots = 4 - len(filtered_feedback)
        if remaining_slots > 0:
            if len(pacing_feedback_sorted) >= min(2, remaining_slots):
                filtered_feedback.extend(pacing_feedback_sorted[:min(2, remaining_slots)])
            else:
                filtered_feedback.extend(pacing_feedback_sorted)

        # P3: Disfluencies (choose 1 or more based on risk score)
        disfluencies_feedback_sorted = sorted(disfluencies_feedback, key=lambda x: x.get('score', 0), reverse=True)
        remaining_slots = 4 - len(filtered_feedback)
        if remaining_slots > 0:
            if len(disfluencies_feedback_sorted) >= remaining_slots:
                filtered_feedback.extend(disfluencies_feedback_sorted[:remaining_slots])
            else:
                filtered_feedback.extend(disfluencies_feedback_sorted)
    
    # De-prioritize showing any feedback on pitch (ensure it's not added)
    final_filtered_feedback = [f for f in filtered_feedback if f.get('type_display') != 'Pitch']
    
    # Ensure max 4 feedbacks
    return final_filtered_feedback[:4]

def show_feedback_container(feedback_list):
    for feedback in feedback_list:
                # Determine score color
                if feedback['score'] <= 3:
                    score_color = '#ff4b4b'  # red
                elif feedback['score'] <= 7:
                    score_color = '#ffd700'  # yellow
                else:
                    score_color = '#4CAF50'  # green
                
                # Create card
                with st.container():
                    st.markdown(f"""
                        <div class="feedback-card">
                            <div class="feedback-header">
                                <div>
                                    You mentioned "{feedback['phrase']}" in the sentence around {(feedback.get('start_time',0))+ feedback.get('end_time',0)//2} seconds.
                                </div>
                                <!-- <div class="feedback-score" style="background-color: {score_color}">
                                    {feedback['score']}
                                </div> -->
                            </div>
                            <div><strong>Feedback:</strong> {feedback['feedback']}</div>
                            <div><strong>Type:</strong> {feedback.get('type_display')}</div>
                            {f'<div class="feedback-tip"><strong>💡 Improvement Suggestion:</strong> {feedback["tip"]}</div>' if feedback.get('tip') else ''}
                        </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

import tempfile
import uuid
import streamlit as st
import os

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
  background-image: url("https://media.istockphoto.com/id/1189302612/photo/multiethnic-specialist-doctors-discussing-case.jpg?s=612x612&w=0&k=20&c=2JwchjpCbkJfyyAfy4CK4lpeZlds6-OCOY4nYPrvwOQ=");
  background-size: cover;
  background-position: center;
}
</style>
'''
page_bg_img = ""

def get_keyword_list():
    # keyword_list = "Quantus 50, Co-enzyme Q10, Selenium, umm, hmm, Udiliv, diabetes, obesity, non-alcoholic liver diseases, liver disease, non-alcoholic fatty liver disease, AST, ALT, GGT, ALP, Ursodeoxycholic acid, position paper endorsed by 4 esteemed societies, Indian society of Gastroenterology, Indian college of cardiology, Endocrine society of India, INASL, cholestasis, hepatoprotective, antioxidant, anti-inflammatory, antiapoptotic, hypercholeretic, Non-alcoholic Liver Disease, 300mg BID, 10-15mg per , kg per day"
    keyword_list = "Quantus 50, Co-enzyme Q10, Selenium,  umm, hmm, uhmm, mmmm, mhmm, uh, uhh, uhm"
    return keyword_list

def load_environment():
    """Load environment variables from .env file"""
    # Load .env file from current directory
    load_dotenv(override=True)
    print("Environment variables loaded successfully!")

def main():
    
    st.set_page_config(layout="wide")
    st.markdown(page_bg_img, unsafe_allow_html=True)
    load_environment()
    feedback_generated = False

    # UI Layout
    col1, col2 = st.columns([1, 2])

    with col1:
        # st.header("Instructions")
        st.write("### Steps to follow:")
        st.write("1. Upload a video or audio file.")
        st.write("2. Upload a transcript text file containing the expected answer.")
        st.write(
            "3. Click the 'Download Feedback' button to get the generated feedback."
        )
        
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

    with col2:
        st.header("Upload Files")

        # File Uploaders
        uploaded_media = st.file_uploader(
            "Upload Video/Audio File", type=["mp4", "mp3", "wav", "avi", "mov"]
        )
        uploaded_text = st.file_uploader("Upload Transcript File", type=["txt"])

        if uploaded_media and uploaded_text:
            if not feedback_generated:
                with st.spinner("Generating feedback... Please wait for a few minutes and do not refresh or close the page."):
                    print(f"File name with ext: {uploaded_media.name}")
                    file_name = uploaded_media.name.split('.')[0]
                    print(f"File name: {file_name}")
                    # Save uploaded media to a temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_media.name)[-1]) as temp_input:
                        temp_input.write(uploaded_media.read())
                        temp_input_path = temp_input.name

                    # Save uploaded text to a temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_media.name)[-1]) as temp_input:
                        temp_input.write(uploaded_media.read())
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
                    non_llm_feedback = filler_detection(word_level_feat_file, silence_txt, window_interval = 0.02)
                    if non_llm_feedback:
                        st.text_area("Filler Words Feedback", value=non_llm_feedback, height=200)
                    ## LLM BASED FEEDBACK
                    prompt_for_audio(aat_file, audio_feedback_file, model=model_option, audio_prompt=audio_prompt)
                    prompt_for_text(ground_truth_path, tt_file, text_feedback_file, model=model_option, text_prompt=text_prompt)
                    prompt_for_quality(ground_truth_path, tt_file, quality_feedback_file, model=model_option, quality_prompt=quality_prompt)
                    all_feedback = collate_all_feedback(tt_file, audio_feedback_file, audio_feedback_file, quality_feedback_file)
                    st.success("Feedback generated successfully!")
                    st.markdown(f"Feedback:\n{all_feedback}")


if __name__ == "__main__":
    main()

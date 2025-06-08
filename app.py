import tempfile
import uuid
import streamlit as st
import os

from feature_processor import word_level_feat_computation
from llm_helper import prompt_for_audio, prompt_for_text, speech_to_text
from speech_helper import get_speech_features, convert_vid_to_audio
from dotenv import load_dotenv

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
    keyword_list = "Quantus 50, Co-enzyme Q10, Selenium, umm, hmm"
    return keyword_list

def load_environment():
    """Load environment variables from .env file"""
    # Load .env file from current directory
    load_dotenv(override=True)
    print(os.getenv("OPENAI_API_KEY"))
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

    with col2:
        st.header("Upload Files")

        # File Uploaders
        uploaded_media = st.file_uploader(
            "Upload Video/Audio File", type=["mp4", "mp3", "wav", "avi", "mov"]
        )
        uploaded_text = st.file_uploader("Upload Transcript File", type=["txt"])

        if uploaded_media and uploaded_text:
            if not feedback_generated:
                with st.spinner("Generating feedback..."):
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
                    word_level_feat_computation(transcription_fl, pitch_txt, energy_txt, silence_txt, word_level_feat_file, aat_file)

                    ## LLM BASED FEEDBACK
                    audio_feedback = prompt_for_audio(aat_file, temp_output_audio_path)
                    corr_fb, quality_fb = prompt_for_text(ground_truth_path, transcription_fl)
                    st.success("Feedback generated successfully!")
                    st.markdown(f"Audio Feedback:\n{audio_feedback}\n\nCorrectness Feedback:\n{corr_fb}\n\n Quality Feedback:\n{quality_fb}")


if __name__ == "__main__":
    main()

import tempfile
import uuid
import streamlit as st
import os

from speech_helper import get_speech_features, convert_vid_to_audio


def main():
    st.set_page_config(layout="wide")
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

        file_id = uuid.uuid4()

        # Placeholder for feedback generation (dummy feedback for now)
        feedback_text = """
        The energy of the user was generally up to the mark, with slight variations in between. 
        Overall, the user was able to maintain a good energy level throughout the video. 
        The pitch of the user was also consistent, with minor fluctuations. 
        The silence detection algorithm identified some silent segments, which could be improved by reducing pauses."""

        if uploaded_media and uploaded_text:
            if not feedback_generated:
                with st.spinner("Generating feedback..."):
                    file_id = uuid.uuid4()

                    # Save uploaded media to a temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_media.name)[-1]) as temp_input:
                        temp_input.write(uploaded_media.read())
                        temp_input_path = temp_input.name

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

                    # Optionally read and display feedback
                    st.success("Feedback generated successfully!")
                    st.text_area("Feedback", feedback_text, height=200)


if __name__ == "__main__":
    main()

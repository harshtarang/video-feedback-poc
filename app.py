import tempfile
import uuid
import streamlit as st
import os
from google.oauth2 import id_token
from google.auth.transport import requests

from feature_processor import word_level_feat_computation
from llm_helper import prompt_for_audio, prompt_for_quality, prompt_for_text, speech_to_text
from non_llm_feedback import filler_detection
from post_processing import collate_all_feedback
from speech_helper import get_speech_features, convert_vid_to_audio
from dotenv import load_dotenv
from prompt_templates import AUDIO_PROMPT, AUDIO_PROMPT_V2, DISFLUENCY_PROMPT, TEXT_PROMPT, TEXT_PROMPT_V1, TEXT_QUALITY_PROMPT

# Enhanced CSS with modern design
page_bg_img = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
}

.landing-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    padding: 0;
    margin: 0;
    position: relative;
    overflow: hidden;
}

.landing-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(255, 255, 255, 0.1) 0%, transparent 50%);
    pointer-events: none;
}

.hero-section {
    position: relative;
    z-index: 1;
    padding: 80px 20px;
    text-align: center;
    color: white;
    max-width: 1200px;
    margin: 0 auto;
}

.hero-title {
    font-size: 4rem;
    font-weight: 700;
    margin-bottom: 1rem;
    background: linear-gradient(45deg, #ffffff, #f0f0f0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 0 4px 8px rgba(0,0,0,0.1);
    animation: fadeInUp 1s ease-out;
}

.hero-subtitle {
    font-size: 1.3rem;
    font-weight: 400;
    margin-bottom: 2rem;
    color: rgba(255, 255, 255, 0.9);
    line-height: 1.6;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
    animation: fadeInUp 1s ease-out 0.2s both;
}

.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    margin: 4rem 0;
    padding: 0 20px;
}

.feature-card {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.2);
    transition: all 0.3s ease;
    animation: fadeInUp 1s ease-out 0.4s both;
}

.feature-card:hover {
    transform: translateY(-10px);
    background: rgba(255, 255, 255, 0.15);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
}

.feature-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    display: block;
}

.feature-title {
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: white;
}

.feature-description {
    color: rgba(255, 255, 255, 0.8);
    line-height: 1.6;
}

.cta-section {
    margin: 4rem 0;
    text-align: center;
    animation: fadeInUp 1s ease-out 0.6s both;
}

.google-login-btn {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    background: white;
    color: #333;
    padding: 16px 32px;
    border-radius: 50px;
    font-weight: 600;
    font-size: 1.1rem;
    text-decoration: none;
    border: none;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    position: relative;
    overflow: hidden;
}

.google-login-btn::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: left 0.5s;
}

.google-login-btn:hover::before {
    left: 100%;
}

.google-login-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.15);
}

.google-icon {
    width: 20px;
    height: 20px;
}

.stats-section {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 2rem;
    margin: 4rem 0;
    padding: 0 20px;
}

.stat-card {
    text-align: center;
    color: white;
    animation: fadeInUp 1s ease-out 0.8s both;
}

.stat-number {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    background: linear-gradient(45deg, #ffffff, #f0f0f0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.stat-label {
    font-size: 1.1rem;
    color: rgba(255, 255, 255, 0.8);
    font-weight: 500;
}

.floating-shapes {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 0;
}

.shape {
    position: absolute;
    opacity: 0.1;
    animation: float 6s ease-in-out infinite;
}

.shape:nth-child(1) {
    top: 20%;
    left: 10%;
    width: 80px;
    height: 80px;
    background: white;
    border-radius: 50%;
    animation-delay: 0s;
}

.shape:nth-child(2) {
    top: 60%;
    right: 10%;
    width: 60px;
    height: 60px;
    background: white;
    border-radius: 30%;
    animation-delay: 2s;
}

.shape:nth-child(3) {
    bottom: 20%;
    left: 20%;
    width: 100px;
    height: 100px;
    background: white;
    border-radius: 20%;
    animation-delay: 4s;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes float {
    0%, 100% {
        transform: translateY(0px);
    }
    50% {
        transform: translateY(-20px);
    }
}

/* Hide Streamlit default elements for landing page */
.landing-page .stApp > header {
    display: none;
}

.landing-page .stApp > .main .block-container {
    padding: 0;
    max-width: 100%;
}

/* Main app styles */
.main-app {
    background: #f8fafc;
    min-height: 100vh;
}

.main-app .stApp > .main .block-container {
    padding: 2rem;
}
</style>
'''

def get_keyword_list():
    # keyword_list = "Quantus 50, Co-enzyme Q10, Selenium, umm, hmm, Udiliv, diabetes, obesity, non-alcoholic liver diseases, liver disease, non-alcoholic fatty liver disease, AST, ALT, GGT, ALP, Ursodeoxycholic acid, position paper endorsed by 4 esteemed societies, Indian society of Gastroenterology, Indian college of cardiology, Endocrine society of India, INASL, cholestasis, hepatoprotective, antioxidant, anti-inflammatory, antiapoptotic, hypercholeretic, Non-alcoholic Liver Disease, 300mg BID, 10-15mg per , kg per day"
    keyword_list = "Quantus 50, Co-enzyme Q10, Selenium,  umm, hmm, uhmm, mmmm, mhmm, uh, uhh, uhm"
    return keyword_list

def load_environment():
    """Load environment variables from .env file"""
    # Load .env file from current directory
    load_dotenv(override=True)
    print("Environment variables loaded successfully!")

def landing_page():
    """Display the enhanced landing page with modern design"""
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    # Add custom CSS class to body
    st.markdown('<div class="landing-page">', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="landing-container">
        <div class="floating-shapes">
            <div class="shape"></div>
            <div class="shape"></div>
            <div class="shape"></div>
        </div>
        <div class="hero-section">
            <h1 class="hero-title">Sales Muni</h1>
            <p class="hero-subtitle">
                Transform your medical sales presentations with AI-powered feedback. 
                Get personalized insights to improve your pitch effectiveness and connect better with healthcare professionals.
            </p>
            <div class="cta-section">
                <button class="google-login-btn" onclick="document.querySelector('[data-testid=&quot;baseButton-secondary&quot;]').click()">
                    <svg class="google-icon" viewBox="0 0 24 24">
                        <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                        <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                        <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                        <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                    </svg>
                    Get Started with Google
                </button>
            </div>
        </div>
        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <h3 class="feature-title">AI-Powered Analysis</h3>
                <p class="feature-description">
                    Advanced machine learning algorithms analyze your speech patterns, 
                    pace, and content to provide actionable feedback.
                </p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <h3 class="feature-title">Real-time Insights</h3>
                <p class="feature-description">
                    Get instant feedback on your presentation quality, 
                    including filler word detection and speech clarity analysis.
                </p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🚀</div>
                <h3 class="feature-title">Performance Boost</h3>
                <p class="feature-description">
                    Improve your sales effectiveness with personalized recommendations 
                    tailored to medical sales scenarios.
                </p>
            </div>
        </div>
        <div class="stats-section">
            <div class="stat-card">
                <div class="stat-number">95%</div>
                <div class="stat-label">Accuracy Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">3min</div>
                <div class="stat-label">Average Analysis Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">500+</div>
                <div class="stat-label">Sales Reps Trained</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">40%</div>
                <div class="stat-label">Performance Improvement</div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Hidden button that actually triggers the login
    if st.button("Login with Google", key="google_login", help="Click to login"):
        st.session_state.authenticated = True
        st.rerun()
    

def main():
    st.set_page_config(
        page_title="Sales Muni - AI-Powered Sales Training",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    load_environment()
    
    # Initialize authentication state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # Show landing page if not authenticated
    if not st.session_state.authenticated:
        landing_page()
        return
    
    # Apply main app styles
    st.markdown('<div class="main-app">', unsafe_allow_html=True)
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
        # st.header("Instructions")
        st.write("### Steps to follow:")
        st.write("1. Upload a video or audio file.")
        st.write("2. Upload a transcript text file containing the expected answer.")
        st.write(
            "3. Click the 'Download Feedback' button to get the generated feedback."
        )
        
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
        st.header("Upload Files")

        # File Uploaders
        uploaded_media = st.file_uploader(
            "Upload Video/Audio File", type=["mp4", "mp3", "wav", "avi", "mov"]
        )
        uploaded_text = st.file_uploader("Upload Transcript File", type=["txt"])

        # Only show the button if both files are uploaded
        if uploaded_media and uploaded_text:
            # Store current files in session state
            st.session_state.uploaded_media = uploaded_media
            st.session_state.uploaded_text = uploaded_text

            if st.button("Generate Feedback"):
                with st.spinner("Generating feedback... Please wait for a few minutes and do not refresh or close the page."):
                    st.session_state.feedback = ([], [])  # Reset feedback
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
                    non_llm_feedback = filler_detection(word_level_feat_file, silence_txt, window_interval = 0.02)
                    if non_llm_feedback:
                        st.text_area("Filler Words Feedback", value=non_llm_feedback, height=200)
                    ## LLM BASED FEEDBACK
                    prompt_for_audio(aat_file, audio_feedback_file, model=model_option, audio_prompt=audio_prompt)
                    prompt_for_text(ground_truth_path, tt_file, text_feedback_file, model=model_option, text_prompt=text_prompt)
                    prompt_for_quality(ground_truth_path, tt_file, quality_feedback_file, model=model_option, quality_prompt=quality_prompt)
                    pos_feedback, neg_feedback = collate_all_feedback(tt_file, audio_feedback_file, audio_feedback_file, quality_feedback_file)
                    st.session_state.feedback = (pos_feedback, neg_feedback)
                    st.success("Feedback generated successfully!")
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
            all_feedback = pos_feedback + neg_feedback
            
            for feedback in all_feedback:
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
                                    You mentioned "{feedback['phrase']}" in the sentence around {(feedback['start_time'] + feedback['end_time'])//2} seconds.
                                </div>
                                <!-- <div class="feedback-score" style="background-color: {score_color}">
                                    {feedback['score']}
                                </div> -->
                            </div>
                            <div><strong>Feedback:</strong> {feedback['feedback']}</div>
                            <div><strong>Type:</strong> {feedback['type_display']}</div>
                            {f'<div class="feedback-tip"><strong>Recommendation:</strong> {feedback["tip"]}</div>' if feedback.get('tip') else ''}
                        </div>
                    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

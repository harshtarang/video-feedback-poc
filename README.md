# Video Feedback POC

A Streamlit application for providing feedback on video transcripts.

## Setup

### Create Virtual Environment

#### Windows
```cmd
python -m venv venv
venv\Scripts\activate
```

#### Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Application

### Normal Mode
```bash
streamlit run app.py
```

### Debug Mode
```bash
# Windows
set DEBUG=1 && streamlit run app.py

# Linux
DEBUG=1 streamlit run app.py
```

## Project Structure
```
├── app.py                 # Main application entry point
├── feature_processor.py   # Feature extraction logic
├── llm_helper.py          # OpenAI API helpers
├── non_llm_feedback.py    # Rule-based feedback
├── openai_utils.py        # OpenAI API utilities
├── prompt_templates.py    # Prompt templates
├── speech_helper.py       # Speech processing utilities
├── utilities.py           # General utilities
├── requirements.txt       # Python dependencies
├── packages.txt           # Additional system packages
└── .streamlit/            # Streamlit configuration
    └── config.toml
```

For more information, refer to the inline documentation in each source file.
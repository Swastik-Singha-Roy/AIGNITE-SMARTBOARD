# AIGNITE SMARTBOARD - EDUBOARD
This project is an interactive AI-powered smartboard and teaching assistant built with Streamlit, integrating voice transcription, Excalidraw whiteboard, generative AI for text and images, and quiz generation features.

---

**Features**

- AI-powered whiteboard (via Excalidraw embed)
- Voice transcription (speech-to-text)
- Meeting/lesson summarization (Gemini API)
- AI-generated quizzes from content
- Image generation (Stability AI, Stable Diffusion 3)
- User-friendly Streamlit web interface

---

## Prerequisites

- **Python 3.8+**
- **pip** (Python package manager)
- API keys for:
    - Google Gemini (set as `GEMINI_API_KEY`)
    - Stability AI (set as `STABILITY_API_KEY`)
- JavaScript
- CSS
- HTML

---

## Installation

1. **Clone the Repository**

```bash
git clone AIGNITE-SMARTBOARD
cd SmartBoard
```

2. **Install Dependencies**

```bash
pip install streamlit speechrecognition google-generativeai pillow requests
```

    - For microphone support (voice transcription), you may also need:
        - `pyaudio` (on Windows: `pip install pyaudio`, on Linux: `sudo apt-get install portaudio19-dev &amp;&amp; pip install pyaudio`)
3. **Set API Keys**

Set your API keys as environment variables:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export STABILITY_API_KEY="your-stability-api-key"
```

Or, create a `.streamlit/secrets.toml` file with:

```toml
GEMINI_API_KEY = "your-gemini-api-key"
STABILITY_API_KEY = "your-stability-api-key"
```


---

## Running the Project

Start the Streamlit app:

```bash
streamlit run app.py
```

- The default browser will open at `http://localhost:8501/`.

---

## Usage

1. **Smartboard Whiteboard**
    - Use the embedded Excalidraw whiteboard for sketches/diagrams.
    - To include your drawing in summaries or quizzes:
        - Open Excalidraw’s menu → “Save as…” → upload the `.excalidraw` file using the uploader.
2. **Teaching Assistant**
    - Click “Start Voice Transcription” to capture spoken notes.
    - Transcripts are listed and can be summarized or used for quiz generation.
3. **Quiz Generation**
    - Click “Generate Quiz from Content” to create multiple-choice questions based on your notes and whiteboard content.
4. **Meeting Summary**
    - Click “Generate Meeting Summary” for an AI-generated summary of the session.
5. **Image Generation**
    - Enter a prompt, select model/aspect ratio, and click “Generate Image” for AI art (Stable Diffusion 3).
6. **Other Controls**
    - “Clear All Notes” to reset transcripts and quizzes.
    - Download generated images directly from the interface.

---

## Troubleshooting

- **Microphone Issues:** Ensure your device has a working microphone and proper permissions.
- **API Errors:** Check that your API keys are set correctly and have sufficient quota.
- **Dependencies:** If you encounter missing package errors, install them via `pip`.

---

## Project Structure

| File/Folder | Purpose |
| :-- | :-- |
| `&lt;main&gt;.py` | Main Streamlit app (see code in paste-2.txt) |
| `requirements.txt` | (Optional) List of dependencies |
| `.streamlit/` | Streamlit config and secrets |
| `index.html` | (Optional) Custom landing page (see paste.txt) |

---

## Credits

- Excalidraw (whiteboard)
- Google Gemini (text AI)
- Stability AI (image generation)
- Streamlit (web app framework)

---

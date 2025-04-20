import streamlit as st
import speech_recognition as sr
import google.generativeai as genai
import os
import json
import requests
from streamlit.components.v1 import html
from PIL import Image
from io import BytesIO

# Configuration
STABILITY_API_HOST = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
STABILITY_MODELS = ["sd3", "sd3-turbo"]

st.set_page_config(page_title="AI Smart Board", layout="wide", initial_sidebar_state="expanded")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def excalidraw_whiteboard():
    return html("""
    <div style="border: 1px solid #ddd; border-radius: 10px; overflow: hidden;">
        <iframe 
            src="https://excalidraw.com/?embed=true"
            style="width: 100%; height: 75vh; border: none;"
            allow="clipboard-read; clipboard-write"
        ></iframe>
    </div>
    """, height=800)

def generate_stable_diffusion_image(prompt, model="sd3", aspect_ratio="1:1", seed=0):
    """Generate image using Stability AI's Stable Diffusion 3 API"""
    try:
        headers = {
            "Authorization": f"Bearer {st.secrets['STABILITY_API_KEY']}",
            "Accept": "image/*"
        }
        
        # Proper multipart/form-data payload
        files = {
            "prompt": (None, prompt),
            "model": (None, model),
            "aspect_ratio": (None, aspect_ratio),
            "seed": (None, str(seed)),
            "output_format": (None, "jpeg")
        }

        response = requests.post(
            "https://api.stability.ai/v2beta/stable-image/generate/sd3",
            headers=headers,
            files=files
        )
        
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Image generation failed: {str(e)}")
        return None


def transcribe_speech(min_confidence=0.80):
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1.2
    with sr.Microphone() as source:
        try:
            st.info("Calibrating mic for background noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            st.info("🎙️ Listening... Please speak clearly.")
            audio = recognizer.listen(source, timeout=3)
            result = recognizer.recognize_google(audio, language="en-IN", show_all=True)
            if result and isinstance(result, dict):
                alt = result.get("alternative", [{}])[0]
                transcript = alt.get("transcript", "").strip()
                confidence = alt.get("confidence", None)
                if transcript:
                    if confidence is not None:
                        if confidence >= min_confidence:
                            return transcript, confidence
                        else:
                            return None, confidence
                    else:
                        return transcript, None
            return None, None
        except sr.WaitTimeoutError:
            return "Mic timeout. Try again.", None
        except sr.UnknownValueError:
            return "Sorry, couldn't understand that.", None
        except sr.RequestError as e:
            return f"Google API error: {e}", None
        except Exception as e:
            return f"Unexpected error: {e}", None

def describe_excalidraw_scene(scene_json):
    try:
        data = json.loads(scene_json)
        elements = data.get("elements", [])
        description = []
        for el in elements:
            el_type = el.get("type")
            text = el.get("text", "")
            if el_type == "text" and text.strip():
                description.append(f"Text: '{text}'")
            elif el_type == "rectangle":
                description.append("Rectangle shape")
            elif el_type == "ellipse":
                description.append("Ellipse shape")
            elif el_type == "arrow":
                description.append("Arrow")
            elif el_type == "diamond":
                description.append("Diamond shape")
            elif el_type == "line":
                description.append("Line")
        if description:
            return "Excalidraw contains: " + "; ".join(description)
        else:
            return "Excalidraw contains no recognizable elements."
    except Exception as e:
        return f"Could not parse Excalidraw JSON: {e}"

def summarize_text(text):
    if not text.strip():
        return "No transcript or whiteboard content to summarize."
    try:
        prompt = (
            "You are an expert assistant summarizing a meeting or discussion.\n\n"
            f"Transcript and Whiteboard Content:\n{text}\n\n"
            "Summarize the main points in 4-5 bullet points, including decisions or action items."
        )
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini Error: {e}"

def safe_parse_quiz(response_text):
    """Robust parser that handles key variations and validates structure"""
    try:
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']')
        if start_idx != -1 and end_idx != -1:
            response_text = response_text[start_idx:end_idx+1]
        questions = json.loads(response_text)
        valid_questions = []
        for q in questions:
            question = q.get('question') or q.get('ques') or q.get('q')
            options = q.get('options') or q.get('opts') or q.get('choices')
            answer = q.get('answer') or q.get('ans') or q.get('correct')
            if not all([question, options, answer]):
                continue
            if isinstance(answer, str):
                answer = answer.strip().upper()[0]
            if len(options) != 4:
                continue
            for i, opt in enumerate(options):
                if not opt.strip().upper().startswith(chr(65+i) + "."):
                    options[i] = f"{chr(65+i)}. {opt.strip()}"
            valid_questions.append({
                'question': question,
                'options': options,
                'answer': answer
            })
        return valid_questions
    except Exception as e:
        st.error(f"Quiz parsing error: {e}")
        return None

def generate_quiz(text, num_questions=5):
    if not text.strip():
        return []
    try:
        prompt = (
            f"Generate {num_questions} multiple-choice questions from this content:\n{text}\n"
            "Format as a JSON list with keys: 'question', 'options' (4 items, each starting with 'A.', 'B.', etc.), 'answer' (A/B/C/D).\n"
            "Example: [{'question':'...','options':['A. ...','B. ...','C. ...','D. ...'],'answer':'A'}]"
        )
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        questions = safe_parse_quiz(response.text)
        if not questions:
            st.error("Failed to generate valid quiz format. Please try again.")
            return []
        return questions[:num_questions]
    except Exception as e:
        st.error(f"Quiz generation failed: {str(e)}")
        return []

def main():
   
    col_whiteboard, col_notes = st.columns([2, 1])

    with col_whiteboard:
        st.header("🖼️ Smartboard Whiteboard")
        excalidraw_whiteboard()
        st.info(
            "To include your drawing in the summary:\n"
            "1. Click the top-left menu in Excalidraw\n"
            "2. Select 'Save as...'\n"
            "3. Upload the file below"
        )
        excalidraw_json = st.file_uploader("Upload your `.excalidraw` file", type=["excalidraw", "json"])
        excalidraw_description = ""
        if excalidraw_json:
            scene_json = excalidraw_json.read().decode("utf-8")
            excalidraw_description = describe_excalidraw_scene(scene_json)
            st.success("Excalidraw scene loaded!")
            st.markdown("**Whiteboard Description:**")
            st.write(excalidraw_description)

    with col_notes:
        st.header("🎤 Teaching Assistant")
        if "transcripts" not in st.session_state:
            st.session_state.transcripts = []

        # Existing Features
        st.markdown("---")
        if st.button("🎤 Start Voice Transcription"):
            text, confidence = transcribe_speech()
            if isinstance(confidence, float):
                if text:
                    st.session_state.transcripts.append(
                        f"{text} (Confidence: {round(confidence * 100, 1)}%)"
                    )
                    st.success(f"✅ Transcribed with {round(confidence * 100, 1)}% confidence: {text}")
                else:
                    st.warning(f"⚠️ Ignored. Confidence too low: {round(confidence * 100, 1)}%")
            elif confidence is None:
                st.session_state.transcripts.append(text)
                st.info(f"✅ Transcribed: {text}")
            else:
                st.error(text)

        if st.button("🧑🏫 Generate Quiz from Content"):
            full_text = " ".join(st.session_state.transcripts)
            if excalidraw_description:
                full_text += "\n\n" + excalidraw_description
            if not full_text.strip():
                st.warning("No content available to generate quiz!")
            else:
                with st.spinner("Generating quiz questions..."):
                    quiz_questions = generate_quiz(full_text)
                if quiz_questions:
                    st.session_state.quiz = {
                        "questions": quiz_questions,
                        "current_q": 0,
                        "score": 0,
                        "user_answers": []
                    }
                    st.success(f"Generated {len(quiz_questions)} questions!")
                else:
                    st.error("Failed to generate valid quiz questions")

        if "quiz" in st.session_state:
            quiz = st.session_state.quiz
            if quiz["current_q"] < len(quiz["questions"]):
                current_q = quiz["questions"][quiz["current_q"]]
                st.markdown("---")
                st.subheader(f"❓ Question {quiz['current_q'] + 1}/{len(quiz['questions'])}")
                st.markdown(f"**{current_q['question']}**")
                user_answer = st.radio(
                    "Select your answer:",
                    options=current_q["options"],
                    index=None,
                    key=f"q_{quiz['current_q']}"
                )
                if st.button("Submit Answer", key=f"submit_{quiz['current_q']}"):
                    if user_answer:
                        quiz["user_answers"].append(user_answer)
                        if user_answer[0].upper() == current_q["answer"]:
                            quiz["score"] += 1
                            st.success("Correct! ✅")
                        else:
                            st.error(f"Wrong! Correct answer: {current_q['answer']} ❌")
                        quiz["current_q"] += 1
                        st.experimental_rerun()
                    else:
                        st.warning("Please select an answer before submitting!")
            else:
                st.markdown("---")
                st.subheader("📊 Quiz Results")
                st.success(f"Your score: {quiz['score']}/{len(quiz['questions'])} ({quiz['score']/len(quiz['questions'])*100:.1f}%)")
                if st.button("🔄 Retake Quiz"):
                    del st.session_state.quiz
                    st.experimental_rerun()

        if st.button("📝 Generate Meeting Summary"):
            full_text = " ".join(st.session_state.transcripts)
            combined_text = full_text
            if excalidraw_description:
                combined_text += "\n\n" + excalidraw_description
            summary = summarize_text(combined_text)
            st.subheader("📌 AI Summary")
            st.write(summary)

        if st.button("🧹 Clear All Notes"):
            st.session_state.transcripts = []
            if "quiz" in st.session_state:
                del st.session_state.quiz
            st.success("✅ All notes cleared!")

        st.subheader("📝 Live Transcripts")
        for i, t in enumerate(st.session_state.transcripts):
            st.markdown(f"**{i+1}.** {t}")
        
          # Image Generation Section
        st.markdown("---")
        st.subheader("🖼️ Image Generation")
        with st.expander("Advanced Settings"):
            model_choice = st.selectbox("Model Version", STABILITY_MODELS, index=0)
            aspect_ratio = st.selectbox("Aspect Ratio", ["1:1", "16:9", "9:16", "3:2", "2:3"], index=0)
            seed = st.number_input("Seed (0 for random)", value=0, min_value=0)
        
        image_prompt = st.text_input("Enter image prompt:", key="sd_prompt")
        if st.button("Generate Image", key="sd_generate"):
            if image_prompt.strip():
                with st.spinner("Generating image..."):
                    generated_image = generate_stable_diffusion_image(
                        prompt=image_prompt,
                        model=model_choice,
                        aspect_ratio=aspect_ratio,
                        seed=seed
                    )
                    if generated_image:
                        st.image(generated_image, caption=image_prompt)
                        st.session_state.transcripts.append(f"Generated image: {image_prompt}")
                        
                        img_bytes = BytesIO()
                        generated_image.save(img_bytes, format="JPEG")
                        st.download_button(
                            label="Download Image",
                            data=img_bytes.getvalue(),
                            file_name="generated_image.jpg",
                            mime="image/jpeg"
                        )
                    else:
                        st.error("Failed to generate image")
            else:
                st.warning("Please enter an image prompt")


if __name__ == "__main__":
    main()

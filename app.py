import streamlit as st
import os
import tempfile
import whisper
from vosk import Model, KaldiRecognizer
import wave
import json
from jiwer import wer, compute_measures

# -----------------------------
# Helpers
# -----------------------------

def transcribe_vosk(wav_path, model_path="vosk-model-small-en-us-0.15"):
    wf = wave.open(wav_path, "rb")
    model = Model(model_path)
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            results.append(json.loads(rec.Result()))
    results.append(json.loads(rec.FinalResult()))
    text = " ".join([r.get("text", "") for r in results]).strip()
    return text

def transcribe_whisper(wav_path, model_size="small"):
    model = whisper.load_model(model_size)
    result = model.transcribe(wav_path)
    return result["text"]

def compute_wer_report(ref_text, hyp_text):
    measures = compute_measures(ref_text, hyp_text)
    return measures, wer(ref_text, hyp_text)

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🎤 Speech-to-Text Evaluation (Milestone 1)")
st.write("Upload an audio file, choose STT engine, and compute WER with reference transcript.")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])
reference_text = st.text_area("Paste Reference Transcript (for WER calculation)", "")

stt_engine = st.radio("Choose STT Engine:", ["Vosk", "Whisper"])

if uploaded_file is not None:
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(tmp_path, format="audio/wav")

    if st.button("Run Transcription"):
        with st.spinner("Transcribing..."):
            if stt_engine == "Vosk":
                # make sure model folder exists
                if not os.path.exists("vosk-model-small-en-us-0.15"):
                    st.error("❌ Vosk model not found. Please download and unzip into project folder.")
                else:
                    transcript = transcribe_vosk(tmp_path, "vosk-model-small-en-us-0.15")
                    st.subheader("Vosk Transcript")
                    st.write(transcript)
            else:
                transcript = transcribe_whisper(tmp_path, "small")
                st.subheader("Whisper Transcript")
                st.write(transcript)

        if reference_text.strip():
            st.subheader("WER Evaluation")
            measures, wer_score = compute_wer_report(reference_text, transcript)
            st.write(f"**WER = {wer_score:.2f}**")
            st.json(measures)

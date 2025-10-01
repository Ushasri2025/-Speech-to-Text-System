import sys
import wave
import json
from vosk import Model, KaldiRecognizer

def transcribe_vosk(audio_path, model_path="vosk-model-small-en-us-0.15"):
    wf = wave.open(audio_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
        raise ValueError("Audio must be WAV format PCM mono.")
    
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

    transcript = " ".join([r.get("text", "") for r in results])
    return transcript.strip()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stt_vosk.py <audio.wav> [model_path]")
        sys.exit(1)

    audio_file = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "vosk-model-small-en-us-0.15"
    print("Transcription (Vosk):")
    print(transcribe_vosk(audio_file, model_path))

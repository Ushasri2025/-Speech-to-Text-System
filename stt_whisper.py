import sys
import whisper

def transcribe_whisper(audio_path, model_size="small"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result["text"].strip()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stt_whisper.py <audio.wav> [model_size]")
        sys.exit(1)

    audio_file = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "small"
    print("Transcription (Whisper):")
    print(transcribe_whisper(audio_file, model_size))

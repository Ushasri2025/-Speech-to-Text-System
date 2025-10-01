import soundfile as sf
import numpy as np
from vosk import Model, KaldiRecognizer
import wave
import json

# Paths
audio_file = "C:/Users/pusha/OneDrive/Desktop/module1/my_audio/my_audio.wav"
model_path = "C:/Users/pusha/OneDrive/Desktop/module1/vosk-model-small-en-us-0.15/vosk-model-small-en-us-0.15"

# Step 1: Load WAV and convert to PCM mono 16kHz
data, samplerate = sf.read(audio_file)
if len(data.shape) > 1:
    data = np.mean(data, axis=1)  # convert to mono

# Resample to 16kHz if needed
if samplerate != 16000:
    from scipy.signal import resample
    num_samples = int(len(data) * 16000 / samplerate)
    data = resample(data, num_samples)
    samplerate = 16000

# Save fixed WAV
sf.write("my_audio_fixed.wav", data, samplerate, subtype='PCM_16')

# Step 2: Transcribe with Vosk
wf = wave.open("my_audio_fixed.wav", "rb")
model = Model(model_path)
rec = KaldiRecognizer(model, wf.getframerate())

result_text = ""
while True:
    frames = wf.readframes(4000)
    if len(frames) == 0:
        break
    if rec.AcceptWaveform(frames):
        res = json.loads(rec.Result())
        result_text += res.get("text", "") + " "

final_res = json.loads(rec.FinalResult())
result_text += final_res.get("text", "")

print("Transcription (Vosk):")
print(result_text.strip())

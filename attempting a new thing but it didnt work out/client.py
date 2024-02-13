import sounddevice as sd
import numpy as np
import requests
from scipy.io.wavfile import write
import io

API_URL = "http://localhost:8756/transcribe"
API_KEY = "your_api_key_here"  # Ensure this matches your actual API key

def record_audio(duration=5, fs=44100, channels=1):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='int16')
    sd.wait()
    print("Recording stopped.")
    return recording, fs

def audio_to_bytesio(audio, fs):
    bytes_io = io.BytesIO()
    write(bytes_io, fs, audio)
    bytes_io.seek(0)  # Go to the beginning of the BytesIO buffer
    return bytes_io

def send_audio_to_server(audio_data):
    files = {'file': ('audio.wav', audio_data, 'audio/wav')}
    headers = {'Authorization': API_KEY}
    response = requests.post(API_URL, files=files, headers=headers)
    return response.json()

def continuously_transcribe():
    try:
        while True:
            audio, fs = record_audio(duration=5)  # Record for 5 seconds
            audio_data = audio_to_bytesio(audio, fs)
            print("Sending audio for transcription...")
            result = send_audio_to_server(audio_data)
            print("Transcription:", result.get('transcription', 'No transcription received.'))
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    continuously_transcribe()

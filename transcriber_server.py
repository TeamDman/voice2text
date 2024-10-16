# server.py

import ssl
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import whisperx
import uvicorn
import io
import traceback
import numpy as np
import soundfile as sf  # Import soundfile for reading audio data
from loguru import logger
import librosa  # For resampling

app = FastAPI()

# Load the WhisperX model
try:
    logger.info("Loading WhisperX model...")
    model = whisperx.load_model("large-v2", device="cuda", language="en")
    logger.info("WhisperX model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load WhisperX model: {e}")
    raise e



def process_audio(audio_data, sample_rate, channel_count):
    # Convert raw byte data into numpy array (assuming float32)
    np_audio = np.frombuffer(audio_data, dtype=np.float32)
    logger.debug(f"Processed audio to numpy array with shape {np_audio.shape}, dtype {np_audio.dtype}")

    # Ensure the length is divisible by the number of channels
    if len(np_audio) % channel_count != 0:
        raise ValueError(f"Audio length {len(np_audio)} is not divisible by channel count {channel_count}")

    # Reshape into (samples_per_channel, channel_count)
    np_audio = np_audio.reshape((-1, channel_count))

    # Extract the desired channel (e.g., right channel for 2-channel audio)
    right_channel = np_audio[:, 1]  # Assuming the right channel is at index 1

    return right_channel


@app.post("/transcribe")
async def transcribe(request: Request):
    # Ensure the Content-Type is audio/wav
    content_type = request.headers.get("Content-Type")
    if content_type != "audio/f32le":
        logger.warning(f"Invalid Content-Type received: {content_type}")
        raise HTTPException(status_code=400, detail="Invalid Content-Type. Expected 'audio/f32le'.")

    # Read the raw bytes from the request body
    try:
        audio_bytes = await request.body()
        logger.debug(f"Received {len(audio_bytes)} bytes of body data.")
    except Exception as e:
        logger.error(f"Failed to read request body: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to read request body: {e}")

    # Convert the raw bytes into a NumPy array of float32
    try:
        np_audio = np.frombuffer(audio_bytes, dtype=np.float32)
        logger.debug(f"Processed audio to numpy array with shape {np_audio.shape}, dtype {np_audio.dtype}")
    except Exception as e:
        logger.error(f"Error processing audio data: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error processing audio data: {e}")

    # Transcribe the audio using WhisperX
    try:
        logger.info("Starting transcription...")
        result = model.transcribe(np_audio, batch_size=16)
        logger.info("Transcription completed successfully.")
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    logger.debug(f"Transcription result: {result}")
    return JSONResponse(content=result)



if __name__ == "__main__":
    ssl_certfile = Path("localhost.pem")
    ssl_keyfile = Path("localhost-key.pem")

    if not ssl_certfile.exists() or not ssl_keyfile.exists():
        logger.error("SSL certificate or key file not found.")
        raise FileNotFoundError("SSL certificate or key file not found.")

    logger.info("Starting server on https://127.0.0.1:8383")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8383,
        ssl_certfile=str(ssl_certfile),
        ssl_keyfile=str(ssl_keyfile),
        log_level="debug",  # Set to 'debug' for more detailed logs
    )

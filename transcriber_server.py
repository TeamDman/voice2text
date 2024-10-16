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
    if content_type != "audio/wav":
        logger.warning(f"Invalid Content-Type received: {content_type}")
        raise HTTPException(status_code=400, detail="Invalid Content-Type. Expected 'audio/wav'.")

    # Read the raw bytes from the request body
    try:
        file_bytes = await request.body()
        logger.debug(f"Received {len(file_bytes)} bytes of body data.")
    except Exception as e:
        logger.error(f"Failed to read request body: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to read request body: {e}")

    # Extract the first 4 bytes as the sample rate (assuming it's a u32 in little-endian format)
    try:
        sample_rate_bytes = file_bytes[:4]
        sample_rate = int.from_bytes(sample_rate_bytes, byteorder="little")
        logger.debug(f"Extracted sample rate: {sample_rate} Hz")

        # Extract the next 2 bytes as the channel count (assuming it's a u16)
        channel_count_bytes = file_bytes[4:6]
        channel_count = int.from_bytes(channel_count_bytes, byteorder="little")
        logger.debug(f"Extracted channel count: {channel_count}")

        # The remaining bytes are the actual raw audio data (in float32 format)
        audio_bytes = file_bytes[6:]
        logger.debug(f"Audio byte length: {len(audio_bytes)}")

        # Ensure the audio byte length is a multiple of 4 (since each f32 is 4 bytes)
        if len(audio_bytes) % 4 != 0:
            raise ValueError(f"Audio byte length {len(audio_bytes)} is not a multiple of 4")

    except Exception as e:
        logger.error(f"Failed to extract metadata or audio data: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to extract metadata or audio data: {e}")

    # Convert the raw bytes into a NumPy array of float32
    try:
        # np_audio = np.frombuffer(audio_bytes, dtype=np.float32)
        # logger.debug(f"Processed audio to numpy array with shape {np_audio.shape}, dtype {np_audio.dtype}")

        # # Ensure the number of samples is divisible by the channel count
        # if len(np_audio) % channel_count != 0:
        #     raise ValueError(f"Audio length {len(np_audio)} is not divisible by channel count {channel_count}")

        # # Handle multiple channels by converting to mono if necessary
        # if channel_count > 1:
        #     logger.info(f"Converting {channel_count}-channel audio to mono.")
        #     np_audio = np.mean(np_audio.reshape(-1, channel_count), axis=1)
        #     logger.debug(f"Mono audio shape: {np_audio.shape}")

        np_audio = process_audio(audio_bytes, sample_rate, channel_count)

        # Resample the audio to 16kHz if necessary
        target_samplerate = 16000  # Whisper models expect 16kHz
        if sample_rate != target_samplerate:
            logger.info(f"Resampling audio from {sample_rate} Hz to {target_samplerate} Hz.")
            np_audio = librosa.resample(np_audio, orig_sr=sample_rate, target_sr=target_samplerate)
            logger.debug(f"Resampled audio shape: {np_audio.shape}")

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

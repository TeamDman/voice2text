import speech_recognition as sr
import asyncio
import torch
import numpy as np
import whisperx
from typing import *

async def record_audio(
    audio_queue: asyncio.Queue[torch.Tensor],
    energy: int,
    pause: float,
    dynamic_energy: bool,
    stop_future: asyncio.Future
):
    r = sr.Recognizer()
    r.energy_threshold = energy
    r.pause_threshold = pause
    r.dynamic_energy_threshold = dynamic_energy

    loop = asyncio.get_running_loop()

    print("[MIC] Starting listener")
    with sr.Microphone(sample_rate=16000) as source:
        print("[MIC] found microphone")
        while not stop_future.done():
            audio = await loop.run_in_executor(None, r.listen, source)
            np_audio = np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0
            # torch_audio = torch.from_numpy(np_audio)
            # print(f"[MIC] Got audio with shape {np_audio.shape}")
            await audio_queue.put(np_audio)
    print("[MIC] Listener finished")

async def transcribe_audio(
    audio_queue: asyncio.Queue[torch.Tensor],
    result_queue: asyncio.Queue[str],
    audio_model: Any,
    stop_future: asyncio.Future
):
    print("[MIC] Starting transcriber")
    while not stop_future.done():
        audio_data = await audio_queue.get()
        result = audio_model.transcribe(audio_data, batch_size=16)
        # predicted_text = str(result["text"]).strip()
        await result_queue.put(result)
    print("[MIC] Transcriber finished")

async def start_background(stop_future: asyncio.Future):
    model = "large-v2"
    audio_model = whisperx.load_model(model, "cuda")

    energy = 100
    pause = 0.8
    dynamic_energy = False

    audio_queue: asyncio.Queue[torch.Tensor] = asyncio.Queue()
    result_queue: asyncio.Queue[str] = asyncio.Queue()

    asyncio.create_task(
        record_audio(
            audio_queue,
            energy,
            pause,
            dynamic_energy,
            stop_future,
        )
    )

    asyncio.create_task(
        transcribe_audio(
            audio_queue,
            result_queue,
            audio_model,
            stop_future,
        )
    )

    return result_queue

async def main():
    stop_future = asyncio.Future()

    result_queue = await start_background(stop_future)

    try:
        while True:
            result = await result_queue.get()
            segments = result["segments"]
            print(segments)
            # print(f"Transcribed text: {result}")
    except KeyboardInterrupt:
        print("Stopping...")
        stop_future.set_result(True)

if __name__ == "__main__":
    asyncio.run(main())

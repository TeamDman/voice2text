import threading
from pynput import keyboard
import pyautogui
import speech_recognition as sr
import asyncio
import torch
import numpy as np
import whisperx
from typing import *

async def record_audio(
    audio_queue: asyncio.Queue[torch.Tensor],
    is_listening: asyncio.Event,
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

    print("[LISTEN] Starting listener")
    with sr.Microphone(sample_rate=16000) as source:
        print("[LISTEN] found microphone")
        while not stop_future.done():
            audio = await loop.run_in_executor(None, r.listen, source)
            if is_listening.is_set():
                print("[LISTEN] Got audio, was listening")
                np_audio = np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0
                await audio_queue.put(np_audio)
            else:
                print("[LISTEN] Got audio, wasn't listening")
    print("[LISTEN] Listener finished")

async def transcribe_audio(
    audio_queue: asyncio.Queue[torch.Tensor],
    result_queue: asyncio.Queue[str],
    audio_model: Any,
    stop_future: asyncio.Future
):
    print("[TRANS] Starting transcriber")
    while not stop_future.done():
        audio_data = await audio_queue.get()
        result = audio_model.transcribe(audio_data, batch_size=16)
        # predicted_text = str(result["text"]).strip()
        print(f"[TRANS] Got result {result}")
        await result_queue.put(result)
    print("[TRANS] Transcriber finished")

async def start_audio_transcription_backend(is_listening: asyncio.Event, stop_future: asyncio.Future):
    model = "large-v2"
    audio_model = whisperx.load_model(model, device="cuda", language="en")

    energy = 100
    pause = 0.8
    dynamic_energy = False

    audio_queue: asyncio.Queue[torch.Tensor] = asyncio.Queue()
    result_queue: asyncio.Queue[str] = asyncio.Queue()

    asyncio.create_task(
        record_audio(
            audio_queue,
            is_listening,
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

async def start_keyboard_backend(is_listening: asyncio.Event, stop_future: asyncio.Future):

    def on_press(key):
        if key == keyboard.Key.f23 and not is_listening.is_set():
            print("[HOTKEY] F23 pressed, starting transcription.")
            is_listening.set()

    def on_release(key):
        if key == keyboard.Key.f23:
            print("[HOTKEY] F23 released, stopping transcription.")
            is_listening.clear()

    def listen_for_keys():
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    listener_thread = threading.Thread(target=listen_for_keys)
    listener_thread.start()

    def stop_check():
        while not stop_future.done():
            threading.Event().wait(0.5)
        listener_thread.join()

    stop_check_thread = threading.Thread(target=stop_check)
    stop_check_thread.start()

        

async def main():
    stop_future = asyncio.Future()
    is_listening = asyncio.Event()

    print("[MAIN] starting audio backend")
    result_queue = await start_audio_transcription_backend(is_listening, stop_future)
    
    print("[MAIN] starting keyboard backend")
    asyncio.create_task(start_keyboard_backend(is_listening, stop_future))

    print("[MAIN] Beginning main loop - hold F23 to perform transcription")
    try:
        while True:
            result = await result_queue.get()
            segments = result["segments"]
            print("[MAIN] Transcribing...", segments)
            to_type = " ".join([segment["text"] for segment in segments]).strip()
            print("[MAIN] Typing...", to_type)
            pyautogui.typewrite(to_type)
    except KeyboardInterrupt:
        print("[MAIN] Stopping...")
        stop_future.set_result(True)

if __name__ == "__main__":
    asyncio.run(main())

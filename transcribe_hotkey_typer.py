import threading
from pynput import keyboard
import pyautogui
import speech_recognition as sr
import asyncio
import torch
import numpy as np
import whisperx
from typing import *
from loguru import logger

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

    logger.info("Starting listener")
    logbackoff = 0
    while not stop_future.done():
        try:
            desired = [(i,v) for i,v in enumerate(sr.Microphone.list_microphone_names()) if v == "Microphone (WOER)"]
            if len(desired) == 0:
                if logbackoff > 0:
                    logbackoff -= 1
                else:
                    logger.info("Desired microphone not found, waiting...")
                    logbackoff = 10
                await asyncio.sleep(1)
                continue
            with sr.Microphone(sample_rate=16000, device_index=desired[0][0]) as source:
                logger.info("Found microphone")
                while not stop_future.done():
                    audio = await loop.run_in_executor(None, r.listen, source)
                    if is_listening.is_set():
                        logger.info("Got audio, was listening")
                        np_audio = np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0
                        await audio_queue.put(np_audio)
                    else:
                        logger.info("Got audio, wasn't listening")
        except OSError:
            logger.exception(f"Microphone error, was it unplugged?")
    logger.info("Listener finished")

async def transcribe_audio(
    audio_queue: asyncio.Queue[torch.Tensor],
    result_queue: asyncio.Queue[str],
    audio_model: Any,
    stop_future: asyncio.Future
):
    logger.info("Starting transcriber")
    while not stop_future.done():
        audio_data = await audio_queue.get()
        result = audio_model.transcribe(audio_data, batch_size=16)
        # predicted_text = str(result["text"]).strip()
        logger.info(f"Got result {result}")
        await result_queue.put(result)
    logger.info("Transcriber finished")

async def start_audio_transcription_backend(is_listening: asyncio.Event, stop_future: asyncio.Future):
    if torch.cuda.is_available():
        model = "large-v2"
        audio_model = whisperx.load_model(model, device="cuda", language="en")
    else:
        model = "small.en"
        audio_model = whisperx.load_model(model, device="cpu", language="en", compute_type="float32")

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
    def is_activation_key(key):
        return key == keyboard.Key.f23 or key == keyboard.Key.pause
    def on_press(key):
        if is_activation_key(key) and not is_listening.is_set():
            logger.info("F23 pressed, starting transcription.")
            is_listening.set()

    def on_release(key):
        if is_activation_key(key):
            logger.info("F23 released, stopping transcription.")
            is_listening.clear()

    def listen_for_keys():
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    listener_thread = threading.Thread(target=listen_for_keys)
    listener_thread.daemon = True
    listener_thread.start()

    def stop_check():
        while not stop_future.done():
            threading.Event().wait(0.5)
        listener_thread.join()

    stop_check_thread = threading.Thread(target=stop_check)
    stop_check_thread.daemon = True
    stop_check_thread.start()

        

async def main():
    stop_future = asyncio.Future()
    is_listening = asyncio.Event()

    logger.info("Starting audio backend")
    result_queue = await start_audio_transcription_backend(is_listening, stop_future)
    
    logger.info("Starting keyboard backend")
    asyncio.create_task(start_keyboard_backend(is_listening, stop_future))

    logger.info("Beginning main loop - hold F23 to perform transcription")
    try:
        while True:
            result = await result_queue.get()
            segments = result["segments"]
            logger.info("Transcribing...", segments)
            to_type = " ".join([segment["text"] for segment in segments]).strip()
            logger.info("Typing...", to_type)
            pyautogui.typewrite(to_type)
    except KeyboardInterrupt:
        logger.info("Stopping...")
        stop_future.set_result(True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Exiting...")
        exit(0)

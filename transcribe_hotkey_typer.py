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
        # https://huggingface.co/openai/whisper-large-v2
        model = "large-v2"
        audio_model = whisperx.load_model(model, device="cuda", language="en")
    else:
        # https://huggingface.co/openai/whisper-small.en
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

async def start_keyboard_backend(
    is_listening: asyncio.Event,
    api_listening: asyncio.Event,
    stop_future: asyncio.Future
):
    def is_activation_key(key):
        return key == keyboard.Key.f23 or key == keyboard.Key.pause

    def on_press(key):
        if is_activation_key(key):
            if not is_listening.is_set():
                logger.info("activation key pressed, starting transcription.")
                is_listening.set()
            api_listening.clear()  # Clear API listening state on manual activation

    def on_release(key):
        if is_activation_key(key):
            logger.info("activation key released, stopping transcription.")
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

        
# async def start_webserver_backend(is_listening: asyncio.Event, stop_future: asyncio.Future, api_key: str):
#     # we want to start a simple web server with https that uses the api_key auth header
#     # we want endpoint for /start_listening that will set is_listening to true
#     # we want endpoint for /stop_listening that will set is_listening to false
#     # we want endpoint for /results that will be a websocket that will send the results of the transcription

import aiohttp.web

async def start_webserver_backend(
    is_listening: asyncio.Event,
    api_listening: asyncio.Event,
    stop_future: asyncio.Future,
    result_queue:asyncio.Queue[torch.Tensor],
    api_key: str
):
    async def start_listening(request):
        # Check API key
        if request.headers.get('Authorization') != api_key:
            return aiohttp.web.Response(status=401, text='Unauthorized')
        
        logger.info("Starting listening because of API request")
        is_listening.set()
        api_listening.set()  # Indicate that listening was started via API
        return aiohttp.web.Response(text="Listening started")

    async def stop_listening(request):
        # Check API key
        if request.headers.get('Authorization') != api_key:
            return aiohttp.web.Response(status=401, text='Unauthorized')
        
        logger.info("Stopping listening because of API request")
        is_listening.clear()
        api_listening.clear()  # Also clear API listening state
        return aiohttp.web.Response(text="Listening stopped")

    async def results(request):
        # Check API key
        if request.headers.get('Authorization') != api_key:
            return aiohttp.web.Response(status=401, text='Unauthorized')
        
        logger.info("Starting websocket for results")
        ws = aiohttp.web.WebSocketResponse()
        await ws.prepare(request)

        while not ws.closed:
            if not result_queue.empty() and api_listening.is_set():
                result = await result_queue.get()
                logger.info(f"Sending result {result}")
                await ws.send_str(str(result))
            else:
                await asyncio.sleep(0.1)
        
        return ws

    app = aiohttp.web.Application()
    app.add_routes([
        aiohttp.web.get('/start_listening', start_listening),
        aiohttp.web.get('/stop_listening', stop_listening),
        aiohttp.web.get('/results', results),
    ])

    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, 'localhost', 8080)
    await site.start()

    # Keep the server running until stop_future is set
    while not stop_future.done():
        await asyncio.sleep(1)

    await runner.cleanup()


async def main():
    import sys
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        api_key = None
    
    stop_future = asyncio.Future()
    is_listening = asyncio.Event()
    api_listening = asyncio.Event()

    logger.info("Starting audio backend")
    result_queue = await start_audio_transcription_backend(is_listening, stop_future)
    
    logger.info("Starting keyboard backend")
    asyncio.create_task(start_keyboard_backend(is_listening, api_listening, stop_future))

    if api_key is not None:
        logger.info("API key supplied, starting web server")
        asyncio.create_task(start_webserver_backend(is_listening, api_listening, stop_future, result_queue, api_key))
    else:
        logger.warn("No API key supplied, not starting web server")

    logger.info("Beginning main loop - hold activation key to perform transcription")
    try:
        while True:
            if api_listening.is_set():
                # Avoid consuming so that the websocket can consume instead
                await asyncio.sleep(1)
                continue
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

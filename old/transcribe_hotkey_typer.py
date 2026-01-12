from loguru import logger
from pynput import keyboard
from typing import *
import aiohttp.web
import asyncio
import json
import numpy as np
import pyautogui
import speech_recognition as sr
import ssl
import threading
import torch
import uuid
import whisperx

async def start_microphone_worker(
    audio_queue: asyncio.Queue[torch.Tensor],
    keyboard_says_listen: asyncio.Event,
    api_says_listen: asyncio.Event,
    energy: int,
    pause: float,
    dynamic_energy: bool,
    stop_future: asyncio.Future
):
    try:
        r = sr.Recognizer()
        r.energy_threshold = energy
        r.pause_threshold = pause
        r.dynamic_energy_threshold = dynamic_energy

        loop = asyncio.get_running_loop()

        logger.info("Starting listener main loop")
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
                        if keyboard_says_listen.is_set() or api_says_listen.is_set():
                            logger.info("Got audio, was listening")
                            np_audio = np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0
                            await audio_queue.put(np_audio)
                        else:
                            logger.info("Got audio, wasn't listening")
            except OSError:
                logger.exception(f"Microphone error, was it unplugged?")
        logger.info("Listener main loop finished")
    except Exception as e:
        logger.exception("Listener main loop crashed: {}", e)

async def start_transcription_worker(
    audio_queue: asyncio.Queue[torch.Tensor],
    result_queue: asyncio.Queue[str],
    stop_future: asyncio.Future
):
    try:
        logger.info("Loading whisperx model")
        try:
            if torch.cuda.is_available():
                # https://huggingface.co/openai/whisper-large-v2
                model = "large-v2"
                audio_model = whisperx.load_model(model, device="cuda", language="en")
            else:
                # https://huggingface.co/openai/whisper-small.en
                model = "small.en"
                audio_model = whisperx.load_model(model, device="cpu", language="en", compute_type="float32")
        except Exception as e:
            logger.exception("Failed to load model: {}", e)
            raise e

        logger.info("Starting transcriber main loop")
        while not stop_future.done():
            audio_data = await audio_queue.get()
            result = audio_model.transcribe(audio_data, batch_size=16)
            # predicted_text = str(result["text"]).strip()
            logger.info(f"Got result {result}")
            await result_queue.put(result)
        logger.info("Transcriber main loop finished")
    except Exception as e:
        logger.exception("Transcriber main loop crashed: {}", e)

async def start_keyboard_worker(
    keyboard_says_listen: asyncio.Event,
    api_says_listen: asyncio.Event,
    stop_future: asyncio.Future
):
    try:
        def is_push_to_talk_key(key):
            return key == keyboard.Key.f23

        def is_toggle_key(key):
            return key == keyboard.Key.pause

        def on_press(key):
            if is_push_to_talk_key(key):
                if not keyboard_says_listen.is_set():
                    logger.info("Push-to-talk key pressed, starting transcription.")
                    keyboard_says_listen.set()
            elif is_toggle_key(key):
                if keyboard_says_listen.is_set():
                    logger.info("Toggle key pressed, stopping transcription.")
                    keyboard_says_listen.clear()
                else:
                    logger.info("Toggle key pressed, starting transcription.")
                    keyboard_says_listen.set()
            api_says_listen.clear()  # Clear API listening state on manual activation

        def on_release(key):
            if is_push_to_talk_key(key) and keyboard_says_listen.is_set():
                logger.info("Push-to-talk key released, stopping transcription.")
                keyboard_says_listen.clear()

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
    except Exception as e:
        logger.exception("Keyboard worker crashed: {}", e)
    
class WebsocketEntry:
    session_id: str
    outbound_queue: asyncio.Queue[str]
    latest_keepalive: float
    def __init__(self, session_id: str, outbound_queue: asyncio.Queue[str]):
        self.session_id = session_id
        self.outbound_queue = outbound_queue
        self.latest_keepalive = asyncio.get_event_loop().time()

active_websockets: dict[str,WebsocketEntry] = {}

async def start_webserver_worker(
    api_says_listen: asyncio.Event,
    stop_future: asyncio.Future,
    port: int,
    api_key: str
):
    try:
        # POST /start_listening
        async def start_listening(request):
            # Check API key
            if request.headers.get('Authorization') != api_key:
                return aiohttp.web.Response(status=401, text='Unauthorized')
            
            logger.info("Starting listening because of API request")
            api_says_listen.set()  # Indicate that listening was started via API
            return aiohttp.web.Response(text="Listening started")

        # POST /stop_listening
        async def stop_listening(request):
            # Check API key
            if request.headers.get('Authorization') != api_key:
                return aiohttp.web.Response(status=401, text='Unauthorized')
            
            logger.info("Stopping listening because of API request")
            api_says_listen.clear()  # Also clear API listening state
            return aiohttp.web.Response(text="Listening stopped")

        # GET /results
        async def results(request):
            # Check API key
            if request.headers.get('Authorization') != api_key:
                return aiohttp.web.Response(status=401, text='Unauthorized')
            
            logger.info("Starting websocket for results")
            return await connect_websocket(request)
        
        # GET /
        async def index(request):
            return aiohttp.web.Response(text="Ahoy!")
        
        async def connect_websocket(request):
            ws = aiohttp.web.WebSocketResponse()
            await ws.prepare(request)

            # Generate a unique ID for each websocket session
            ws_queue = asyncio.Queue()

            global active_websockets
            entry = WebsocketEntry(session_id=str(uuid.uuid4()), outbound_queue=ws_queue)
            active_websockets[entry.session_id] = entry

            logger.info("Starting keepalive receiver")
            asyncio.create_task(start_keepalive_receiver(ws, entry, stop_future))

            try:
                while not ws.closed and not stop_future.done():
                    to_send = await ws_queue.get()
                    await ws.send_str(json.dumps(to_send))
            finally:
                # for some reason we never get here
                # the keepalive receiver does trigger closed tho
                # and if that fails, the keepalive will clean it up
                logger.info("Websocket closed")
                del active_websockets[entry.session_id]

            return ws
        
        async def start_keepalive_receiver(ws: aiohttp.web.WebSocketResponse, entry: WebsocketEntry, stop_future: asyncio.Future):
            while not ws.closed and not stop_future.done():
                message = await ws.receive()
                if message.type == aiohttp.WSMsgType.TEXT:
                    content = message.data
                    if content == "keepalive":
                        logger.debug("Received keepalive from {}", entry.session_id)
                        entry.latest_keepalive = asyncio.get_event_loop().time()
                    else:
                        logger.warning(f"Unknown message received: {content}")
                elif message.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"Websocket error: {ws.exception()}")
                    break
            logger.info("Websocket closed")
            del active_websockets[entry.session_id]
        
        async def start_keepalive_prune_worker(stop_future: asyncio.Future):
            global active_websockets
            timeout = 10
            while not stop_future.done():
                await asyncio.sleep(10)
                for session_id, entry in active_websockets.items():
                    if asyncio.get_event_loop().time() - entry.latest_keepalive > timeout:
                        logger.info(f"No keepalive received from session {session_id} in the past {timeout} seconds, closing")
                        del active_websockets[session_id]

        async def start_auto_unlisten_worker(stop_future: asyncio.Future):
            global active_websockets
            while not stop_future.done():
                await asyncio.sleep(1)
                if api_says_listen.is_set() and len(active_websockets) == 0:
                    logger.info("No active websockets, unsetting api_listening")
                    api_says_listen.clear()

        logger.info("Starting keepalive prune worker")
        asyncio.create_task(start_keepalive_prune_worker(stop_future))
        logger.info("Starting auto unlisten worker")
        asyncio.create_task(start_auto_unlisten_worker(stop_future))

        app = aiohttp.web.Application()
        app.add_routes([
            aiohttp.web.post('/start_listening', start_listening),
            aiohttp.web.post('/stop_listening', stop_listening),
            aiohttp.web.get('/results', results),
            aiohttp.web.get('/', index),
        ])

        runner = aiohttp.web.AppRunner(app)
        await runner.setup()

        
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain("localhost.pem", "localhost-key.pem")

        logger.info(f"Starting web server on port {port}")
        site = aiohttp.web.TCPSite(runner, "localhost", port, ssl_context=ssl_context)
        await site.start()

        # Keep the server running until stop_future is set
        while not stop_future.done():
            await asyncio.sleep(1)

        await runner.cleanup()
    except Exception as e:
        logger.exception("Webserver worker crashed: {}", e)


async def start_router_worker(api_says_listen: asyncio.Event, result_queue: asyncio.Queue[str], typewriter_queue: asyncio.Queue[str], stop_future: asyncio.Future):
    try:
        global active_websockets
        while not stop_future.done():
            result = await result_queue.get()
            if api_says_listen.is_set():
                # Forward result to each websocket queue
                for entry in active_websockets.values():
                    await entry.outbound_queue.put(result)
            else:
                # Forward result to the typewriter queue
                await typewriter_queue.put(result)
    except Exception as e:
        logger.exception("Router worker crashed: {}", e)
            
async def main():
    try:
        import sys
        import os
        
        port = os.getenv('port')
        api_key = os.getenv('api_key')


        # if len(sys.argv) > 2:
        #     port = sys.argv[1]
        #     api_key = sys.argv[2]
        # else:
        #     api_key = None
        
        stop_future = asyncio.Future()
        keyboard_says_listen = asyncio.Event()
        api_says_listen = asyncio.Event()

        audio_queue: asyncio.Queue[torch.Tensor] = asyncio.Queue()
        result_queue: asyncio.Queue[str] = asyncio.Queue()
        typewriter_queue: asyncio.Queue[str] = asyncio.Queue()

        logger.info("Starting microphone worker")
        asyncio.create_task(
            start_microphone_worker(
                audio_queue,
                keyboard_says_listen,
                api_says_listen,
                energy=300,
                pause=0.8,
                dynamic_energy=False,
                stop_future=stop_future,
            )
        )

        logger.info("Starting transcription worker")
        asyncio.create_task(start_transcription_worker(audio_queue, result_queue, stop_future))

        logger.info("Starting keyboard worker")
        asyncio.create_task(start_keyboard_worker(keyboard_says_listen, api_says_listen, stop_future))

        logger.info("Starting background router")
        asyncio.create_task(start_router_worker(api_says_listen, result_queue, typewriter_queue, stop_future))


        if api_key is not None:
            logger.info("API key supplied, starting webserver worker")
            asyncio.create_task(start_webserver_worker(api_says_listen, stop_future, port, api_key))
        else:
            logger.warning("No API key supplied, not starting web server")

        logger.info("Beginning main loop - hold activation key to perform transcription")
        try:
            while True:
                result = await typewriter_queue.get()
                segments = result["segments"]
                logger.info("Transcribing...", segments)
                to_type = " ".join([segment["text"] for segment in segments]).strip()
                logger.info("Typing...", to_type)
                pyautogui.typewrite(to_type)
        except KeyboardInterrupt:
            logger.info("Stopping...")
            stop_future.set_result(True)
    except Exception as e:
        logger.exception("Main loop crashed: {}", e)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Exiting...")
        exit(0)

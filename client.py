import aiohttp
import asyncio
import sys

from concurrent.futures import ThreadPoolExecutor

API_KEY = 'abc123'
SERVER_URL = 'http://localhost:8080'

async def read_input(executor):
    loop = asyncio.get_running_loop()
    while True:
        # Run input in a separate thread
        command = await loop.run_in_executor(executor, input, "Enter command (start, stop, exit): ")
        command = command.strip().lower()
        if command == "exit":
            print("Exiting...")
            break
        elif command in ["start", "stop"]:
            await send_command(command)
        else:
            print("Unknown command.")

async def send_command(command):
    url = f"{SERVER_URL}/{command}_listening"
    headers = {'Authorization': API_KEY}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            print(f"Command '{command}': {response.status}")
            print(await response.text())

async def receive_results():
    url = f"{SERVER_URL}/results"
    headers = {'Authorization': API_KEY}
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url, headers=headers) as ws:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    print("Received:", msg.data)
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break

async def main():
    executor = ThreadPoolExecutor(max_workers=1)
    # Start the results receiver in the background
    receiver_task = asyncio.create_task(receive_results())
    # Start the input reader in the background
    input_task = asyncio.create_task(read_input(executor))

    # Wait for the input_task to complete, indicating the user has chosen to exit
    await input_task
    
    # Cleanup
    receiver_task.cancel()
    try:
        await receiver_task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    if len(sys.argv) > 1:
        API_KEY = sys.argv[1]
    asyncio.run(main())

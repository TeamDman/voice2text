import aiohttp
import asyncio
import sys

API_KEY = 'abc123'
SERVER_URL = 'http://localhost:8080'

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
    # Start the results receiver in the background
    receiver_task = asyncio.create_task(receive_results())

    # Command input loop
    try:
        while True:
            command = input("Enter command (start, stop, exit): ").strip().lower()
            if command == "exit":
                break
            elif command in ["start", "stop"]:
                await send_command(command)
            else:
                print("Unknown command.")
    except KeyboardInterrupt:
        print("Exiting...")

    receiver_task.cancel()
    try:
        await receiver_task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    if len(sys.argv) > 1:
        API_KEY = sys.argv[1]
    asyncio.run(main())

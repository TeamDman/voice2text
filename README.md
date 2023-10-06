# voice2text
A python program that types what you say if you're holding the hotkey


## Installation

Install [whisperx](https://github.com/m-bain/whisperX), which creates a new conda environment

Install our dependencies

```pwsh
pip install pyautogui pynput pyaudio speechrecognition sounddevice pydub loguru
```

## Usage

```pwsh
python .\transcribe_hotkey_typer.py
```

You need to hold the hotkey until it types. Default is F23, which I have bound to a mouse button.

When you hit Ctrl+C, the program will wait for you to say something before the voice thread will exit.
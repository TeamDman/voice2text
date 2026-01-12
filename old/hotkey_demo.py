from pynput import keyboard

def on_key_event(key, action):
    try:
        print(f"Key {key.char} is {action}")
    except AttributeError:
        print(f"Special key {key} is {action}")

def on_press(key):
    on_key_event(key, "pressed")

def on_release(key):
    on_key_event(key, "released")
    if key == keyboard.Key.esc:
        # Stop listener when escape key is pressed
        return False

def main():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

if __name__ == "__main__":
    main()

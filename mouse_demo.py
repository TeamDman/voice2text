from pynput import mouse

def on_click(x, y, button, pressed):
    action = "pressed" if pressed else "released"
    print(f"Button {button} is {action} at ({x}, {y})")

def main():
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()

if __name__ == "__main__":
    main()

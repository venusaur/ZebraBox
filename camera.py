import cv2
import tkinter as tk
from tkinter import ttk
import os
import time
import threading

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera App")
        
        # Capture button to manually take in image
        self.capture_button = ttk.Button(self.root, text="Capture", command=self.capture_image)
        self.capture_button.pack(pady=10)


        # Field of entering path for images
        self.folder_path = ttk.Entry(self.root, width=40)
        self.folder_path.insert(0, os.path.abspath("captured_images"))
        self.folder_path.pack(pady=5)
        
        # Field for duration of capturing
        self.capture_duration_label = ttk.Label(self.root, text="Capture Duration (seconds):")
        self.capture_duration_label.pack(pady=5)
        
        self.capture_duration = ttk.Entry(self.root)
        self.capture_duration.insert(0, "10")  # Default capture duration is 10 seconds
        self.capture_duration.pack(pady=5)

        self.start_button = ttk.Button(self.root, text="Start", command=self.start_capture)
        self.start_button.pack(pady=10)

        self.stop_button = ttk.Button(self.root, text="Stop", command=self.stop_capture)
        self.stop_button.pack(pady=10)
        self.stop_button.config(state="disabled")

        self.capture = cv2.VideoCapture(0)
        self.is_capturing = False
    

    def capture_image(self):
        if not os.path.exists(self.folder_path.get()):
            os.makedirs(self.folder_path.get())

        ret, frame = self.capture.read()
        timestamp = int(time.time())
        image_path = os.path.join(self.folder_path.get(), f"capture_{timestamp}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Image saved to {image_path}")

    def start_capture(self):
        self.is_capturing = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")

        capture_duration = float(self.capture_duration.get())
        # capture_duration = float(self.capture_duration.get()) / 1000.0 # convert to milliseconds

        self.capture_thread = threading.Thread(target=self.capture_loop, args=(capture_duration,))
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def stop_capture(self):
        self.is_capturing = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

    def capture_loop(self, capture_duration):
        start_time = time.time()
        while self.is_capturing and time.time() - start_time <= capture_duration:
            self.capture_image()
            time.sleep(1)
            # time.sleep(0.001) // milliseconds

def main():
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()


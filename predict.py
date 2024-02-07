from ultralytics import YOLO
import mss
from PIL import Image
import ctypes
import numpy as np
import time
import interception
import mouse
import winsound
import subprocess
import asyncio
import cv2


async def move_mouse(x, y):
    # Move the mouse asynchronously
    interception.move_relative(int(x), int(y))

# Check if ran as admin
if not subprocess.check_output('net session', shell=True):
    print("Run as admin")
    exit(0)

# Load a model
interception.auto_capture_devices(keyboard=True, mouse=True)
model = YOLO('best.pt').cuda()  # pretrained YOLOv8n model, move it to GPU

# Start
winsound.PlaySound("C:\Windows\Media\Speech On.wav", winsound.SND_FILENAME)

# Clear the terminal using the cls command
subprocess.call("cls", shell=True)

import mss.tools

# Define the screen dimensions
screen_width = ctypes.windll.user32.GetSystemMetrics(0)
screen_height = ctypes.windll.user32.GetSystemMetrics(1)

# Define the relative coordinates for capturing the image
capture_width = 416
capture_height = 416

# Define the field of view (FOV) dimensions
FOV_width = 150
FOV_height = 150

# Define the speed of the mouse
speed = 1.25

# Calculate the FOV position
FOV_x = (screen_width - FOV_width) // 2
FOV_y = (screen_height - FOV_height) // 2

async def main():
    while True:
        # Capture the FOV region
        with mss.mss() as sct:
            monitor = {"top": FOV_y, "left": FOV_x, "width": FOV_width, "height": FOV_height}
            screenshot = sct.grab(monitor)
            # Convert the screenshot to a PIL image
            screenshot_np = np.array(screenshot)
            screenshot_np = cv2.cvtColor(screenshot_np, cv2.COLOR_RGBA2RGB)
            screenshot_np = cv2.resize(screenshot_np, (capture_width, capture_height))

        results = model.predict(screenshot_np, stream=True, device="0", verbose=False, classes=1)

        if results:
            closest_box_distance = float('inf')
            closest_box_center = None

            for r in results:
                if r.boxes.xyxy.shape[0] == 0:
                    continue
                # if the boxes are not empty
                else:
                    # get the x and y values of the center of the bounding box, it can be multiple boxes
                    for i in range(r.boxes.xyxy.shape[0]):
                        x = int((r.boxes.xyxy[i][0] + r.boxes.xyxy[i][2]) / 2)
                        y = int((r.boxes.xyxy[i][1] + r.boxes.xyxy[i][3]) / 2)

                        # Convert the tensor value to a numerical value
                        x = int(x)
                        y = int(y)

                        # Scale the coordinates back up to the screen scale
                        x = x * FOV_width // capture_width + FOV_x
                        y = y * FOV_height // capture_height + FOV_y

                        # Calculate the distance from the center of the screen
                        distance = np.sqrt((x - screen_width / 2) ** 2 + (y - screen_height / 2) ** 2)

                        # Update the closest box if the current box is closer
                        if distance < closest_box_distance:
                            closest_box_distance = distance
                            closest_box_center = (x, y)

            if closest_box_center and not mouse.is_pressed(button='x'):
                # Move the mouse towards the closest box
                relative_x = closest_box_center[0] - screen_width / 2
                relative_y = closest_box_center[1] - screen_height / 2
                await move_mouse(relative_x * speed, relative_y * speed)

asyncio.run(main())
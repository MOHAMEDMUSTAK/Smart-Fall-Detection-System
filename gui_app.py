import torch
import cv2
import time
import os
import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk
from torchvision import transforms, models
import torch.nn as nn
import pygame

# -------------------------
# SPEED OPTIMIZATION
# -------------------------
torch.set_num_threads(2)
device = torch.device("cpu")

# -------------------------
# INIT SOUND
# -------------------------
pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3")

# -------------------------
# LOAD MODEL
# -------------------------
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load("fall_model.pth", map_location=device))
model = model.to(device)
model.eval()

classes = ["fall", "not_fall"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

if not os.path.exists("alerts"):
    os.makedirs("alerts")

# -------------------------
# GUI SETUP
# -------------------------
root = tk.Tk()
root.title("Smart Fall Detection System")
root.geometry("1000x700")
root.configure(bg="#111827")

title = Label(root,
              text="SMART FALL DETECTION SYSTEM",
              font=("Arial", 22, "bold"),
              bg="#111827",
              fg="#00FFD1")
title.pack(pady=15)

video_label = Label(root)
video_label.pack()

status_label = Label(root,
                     text="Status: SAFE",
                     font=("Arial", 18, "bold"),
                     bg="#111827",
                     fg="green")
status_label.pack(pady=10)

alert_count_label = Label(root,
                          text="Alerts: 0",
                          font=("Arial", 16),
                          bg="#111827",
                          fg="white")
alert_count_label.pack()

button_frame = Frame(root, bg="#111827")
button_frame.pack(pady=20)

# -------------------------
# VARIABLES
# -------------------------
cap = None
running = False
alert_count = 0

fall_frame_count = 0
fall_confirm_threshold = 6
alarm_playing = False

frame_counter = 0
current_label = "not_fall"

# -------------------------
# CAMERA FUNCTIONS
# -------------------------
def start_camera():
    global cap, running
    if not running:
        cap = cv2.VideoCapture(0)
        running = True
        update_frame()

def stop_camera():
    global running
    running = False
    if cap:
        cap.release()
    pygame.mixer.music.stop()

# -------------------------
# UPDATE FRAME
# -------------------------
def update_frame():
    global fall_frame_count, alarm_playing
    global alert_count, frame_counter, current_label

    if running:
        ret, frame = cap.read()
        if ret:

            frame_counter += 1

            # Only predict every 5 frames (FAST)
            if frame_counter % 5 == 0:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                input_tensor = transform(pil_img).unsqueeze(0).to(device)

                with torch.inference_mode():
                    output = model(input_tensor)
                    _, pred = torch.max(output, 1)

                current_label = classes[pred.item()]

            # Multi-frame confirmation
            if current_label == "fall":
                fall_frame_count += 1
            else:
                fall_frame_count = 0

            if fall_frame_count >= fall_confirm_threshold:
                status_label.config(text="Status: FALL DETECTED",
                                    fg="red")

                if not alarm_playing:
                    pygame.mixer.music.play(-1)
                    alarm_playing = True

                if fall_frame_count == fall_confirm_threshold:
                    alert_count += 1
                    alert_count_label.config(text=f"Alerts: {alert_count}")
                    filename = f"alerts/fall_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)

            else:
                status_label.config(text="Status: SAFE",
                                    fg="green")

                if alarm_playing:
                    pygame.mixer.music.stop()
                    alarm_playing = False

            display_frame = cv2.resize(frame, (800, 450))
            display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(display_rgb))

            video_label.imgtk = img
            video_label.configure(image=img)

        video_label.after(15, update_frame)

# -------------------------
# BUTTONS
# -------------------------
start_button = Button(button_frame,
                      text="Start",
                      font=("Arial", 14),
                      command=start_camera,
                      bg="#16a34a",
                      fg="white",
                      width=12)

stop_button = Button(button_frame,
                     text="Stop",
                     font=("Arial", 14),
                     command=stop_camera,
                     bg="#dc2626",
                     fg="white",
                     width=12)

start_button.grid(row=0, column=0, padx=30)
stop_button.grid(row=0, column=1, padx=30)

root.mainloop()

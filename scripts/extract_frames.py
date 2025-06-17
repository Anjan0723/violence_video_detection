import os
import cv2
import numpy as np

VIDEO_DIR = "data"
FRAME_DIR = "frames"
FRAME_SIZE = (128, 128)
SKIP_FRAMES = 7

def extract_and_save_frames():
    categories = ['Violence', 'NonViolence']
    os.makedirs(FRAME_DIR, exist_ok=True)

    for category in categories:
        video_folder = os.path.join(VIDEO_DIR, category)
        frame_folder = os.path.join(FRAME_DIR, category)
        os.makedirs(frame_folder, exist_ok=True)

        for filename in os.listdir(video_folder):
            if not filename.endswith(".mp4"):
                continue
            video_path = os.path.join(video_folder, filename)
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            saved_frames = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % SKIP_FRAMES == 0:
                    frame = cv2.resize(frame, FRAME_SIZE)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame.astype('float32') / 255.0
                    save_path = os.path.join(frame_folder, f"{filename}_{saved_frames}.npy")
                    np.save(save_path, frame)
                    saved_frames += 1

                frame_count += 1

            cap.release()

if __name__== "__main__":
    extract_and_save_frames()
    print("[INFO] Frame extraction completed.")
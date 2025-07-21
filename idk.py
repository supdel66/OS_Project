import cv2
import mediapipe as mp
import os
import time
from glob import glob

MEMES_DIR = "memes"
SLIDES_DIR = "slides"
SWITCH_DELAY = 0.1  # seconds

meme_imgs = sorted(glob(os.path.join(MEMES_DIR, "*.jpg")) + glob(os.path.join(MEMES_DIR, "*.png")))
slide_imgs = sorted(glob(os.path.join(SLIDES_DIR, "*.jpg")) + glob(os.path.join(SLIDES_DIR, "*.png")))
slide_idx = 0
meme_idx = 0  # Add this line to track current meme index

if not meme_imgs:
    raise RuntimeError("No meme images found in 'memes/' folder!")
if not slide_imgs:
    raise RuntimeError("No slide images found in 'slides/' folder!")

cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

last_state = None
last_switch = time.time()

def is_facing_camera(landmarks):
    nose = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    eye_center = (left_eye.x + right_eye.x) / 2
    return 0.35 < eye_center < 0.65 and 0.4 < nose.x < 0.6

# Preload images
meme_imgs_cv = [cv2.imread(img) for img in meme_imgs]  # Preload all memes
slide_imgs_cv = [cv2.imread(img) for img in slide_imgs]

cv2.namedWindow("Display", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    facing = False
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            if is_facing_camera(face_landmarks.landmark):
                facing = True
                break

    now = time.time()
    if facing != last_state and now - last_switch > SWITCH_DELAY:
        if facing:
            print(f"Showing MEME: {meme_imgs[meme_idx]}")
            display_img = meme_imgs_cv[meme_idx]
            meme_idx = (meme_idx + 1) % len(meme_imgs_cv)  # Cycle to next meme
        else:
            print(f"Showing SLIDE: {slide_imgs[slide_idx]}")
            display_img = slide_imgs_cv[slide_idx]
            slide_idx = (slide_idx + 1) % len(slide_imgs_cv)
        last_state = facing
        last_switch = now

        # Resize to fit screen if needed
        screen_res = (1920, 1080)  # Change to your projector/screen resolution
        display_img_resized = cv2.resize(display_img, screen_res, interpolation=cv2.INTER_AREA)
        cv2.imshow("Display", display_img_resized)

    # Optional: show camera preview in a small window
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
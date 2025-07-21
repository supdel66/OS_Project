import cv2
import mediapipe as mp
import os
import time
import argparse
from glob import glob
import numpy as np

# Add argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--camera', type=str, default='0',
                   help='Camera source: 0 for webcam or URL for IP camera')
args = parser.parse_args()

MEMES_DIR = "memes"
SLIDES_DIR = "slides"
SWITCH_DELAY = 0.2  # Slightly increased to reduce rapid switches

meme_imgs = sorted(glob(os.path.join(MEMES_DIR, "*.jpg")) + glob(os.path.join(MEMES_DIR, "*.png")))
slide_imgs = sorted(glob(os.path.join(SLIDES_DIR, "*.jpg")) + glob(os.path.join(SLIDES_DIR, "*.png")))
slide_idx = 0
meme_idx = 0  # Add this line to track current meme index

if not meme_imgs:
    raise RuntimeError("No meme images found in 'memes/' folder!")
if not slide_imgs:
    raise RuntimeError("No slide images found in 'slides/' folder!")

# Modified camera initialization
if args.camera.isdigit():
    cap = cv2.VideoCapture(int(args.camera))
else:
    # For IP camera, append '/video' to the URL if not present
    camera_url = args.camera
    if not camera_url.endswith('/video'):
        camera_url += '/video'
    cap = cv2.VideoCapture(camera_url)
    
if not cap.isOpened():
    raise RuntimeError("Failed to open camera source!")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.4,  # Reduced for better performance
    min_tracking_confidence=0.4,
    refine_landmarks=False  # Disable landmark refinement for speed
)

last_state = None
last_switch = time.time()

def is_facing_camera(landmarks):
    nose = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    eye_center = (left_eye.x + right_eye.x) / 2
    return 0.35 < eye_center < 0.65 and 0.4 < nose.x < 0.6

# Cache the screen resolution
screen_res = (1920, 1080)  # Adjust to your screen resolution

# Preload and resize images to screen resolution
meme_imgs_cv = [cv2.resize(cv2.imread(img), screen_res) for img in meme_imgs]
slide_imgs_cv = [cv2.resize(cv2.imread(img), screen_res) for img in slide_imgs]

# Add these optimized constants
PROCESS_FRAME_SIZE = (320, 240)  # Smaller size for processing
PREVIEW_SIZE = (240, 180)  # Smaller preview window
PREVIEW_PADDING = 10
OVERLAY_ALPHA = 0.7

# Modified main loop for better performance
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Process smaller frame for face detection
    frame_small = cv2.resize(frame, PROCESS_FRAME_SIZE)
    rgb_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_small)
    
    facing = False
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]  # Only check first face
        if is_facing_camera(face_landmarks.landmark):
            facing = True

    now = time.time()
    if facing != last_state and now - last_switch > SWITCH_DELAY:
        if facing:
            display_img = meme_imgs_cv[meme_idx]
            meme_idx = (meme_idx + 1) % len(meme_imgs_cv)
        else:
            display_img = slide_imgs_cv[slide_idx]
            slide_idx = (slide_idx + 1) % len(slide_imgs_cv)
        last_state = facing
        last_switch = now

        # Create efficient main display
        main_display = display_img.copy()
        
        # Add smaller preview
        preview = cv2.resize(frame, PREVIEW_SIZE)
        preview_y = PREVIEW_PADDING
        preview_x = main_display.shape[1] - PREVIEW_SIZE[0] - PREVIEW_PADDING
        
        # Simplified overlay
        cv2.rectangle(main_display, 
                     (preview_x - 5, preview_y - 5),
                     (preview_x + PREVIEW_SIZE[0] + 5, preview_y + PREVIEW_SIZE[1] + 5),
                     BG_COLOR, -1)
        
        # Add preview directly
        main_display[preview_y:preview_y + PREVIEW_SIZE[1],
                    preview_x:preview_x + PREVIEW_SIZE[0]] = preview

        # Simplified status display
        status_text = "Face" if facing else "No Face"
        cv2.putText(main_display, status_text,
                    (preview_x, preview_y + PREVIEW_SIZE[1] + 20),
                    FONT, 0.6, TEXT_COLOR, 1)

        cv2.imshow(WINDOW_NAME, main_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
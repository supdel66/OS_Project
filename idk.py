import cv2
import mediapipe as mp
import os
import time
import threading
import argparse
from glob import glob

# Add argument parser at the top
parser = argparse.ArgumentParser()
parser.add_argument('--camera', type=str, default='0',
                   help='Camera URL (e.g., http://192.168.0.102:8080/video)')
args = parser.parse_args()

MEMES_DIR = "memes"
SLIDES_DIR = "slides"
SWITCH_DELAY = 0.1
PREVIEW_WIDTH = 320  # Preview window width
PREVIEW_HEIGHT = 240  # Preview window height

# Load images
meme_imgs = sorted(glob(os.path.join(MEMES_DIR, "*.jpg")) + glob(os.path.join(MEMES_DIR, "*.png")))
slide_imgs = sorted(glob(os.path.join(SLIDES_DIR, "*.jpg")) + glob(os.path.join(SLIDES_DIR, "*.png")))

slide_idx = 0
meme_idx = 0

if not meme_imgs:
    raise RuntimeError("No meme images found in 'memes/' folder!")
if not slide_imgs:
    raise RuntimeError("No slide images found in 'slides/' folder!")

# Preload images
print(f"Loading {len(meme_imgs)} memes and {len(slide_imgs)} slides...")
meme_imgs_cv = [cv2.imread(img) for img in meme_imgs]
slide_imgs_cv = [cv2.imread(img) for img in slide_imgs]
print("✅ Images loaded!")

# Replace the camera initialization
if args.camera.isdigit():
    cap = cv2.VideoCapture(int(args.camera))
else:
    # For IP webcam
    cap = cv2.VideoCapture(args.camera)
    
if not cap.isOpened():
    raise RuntimeError("Failed to open camera! Check IP address or connection.")

mp_face_mesh = mp.solutions.face_mesh
# Optimize face detection for performance
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=False
)

last_state = None
last_switch = time.time()

def is_facing_camera(landmarks):
    nose = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    eye_center = (left_eye.x + right_eye.x) / 2
    return 0.35 < eye_center < 0.65 and 0.4 < nose.x < 0.6

def print_status():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 60)
    print("🎭 FACE DETECTION PRESENTATION")
    print("=" * 60)
    print(f"📊 Memes: {meme_idx}/{len(meme_imgs)} | Slides: {slide_idx}/{len(slide_imgs)}")
    print("🟢 RUNNING - Press 'q' in camera window to quit")
    print("=" * 60)

cv2.namedWindow("Display", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Status update thread
def status_updater():
    while True:
        print_status()
        time.sleep(2)

status_thread = threading.Thread(target=status_updater, daemon=True)
status_thread.start()

print_status()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Resize camera preview to smaller size
    preview_frame = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))

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
            display_img = meme_imgs_cv[meme_idx]
            meme_idx = (meme_idx + 1) % len(meme_imgs_cv)
        else:
            display_img = slide_imgs_cv[slide_idx]
            slide_idx = (slide_idx + 1) % len(slide_imgs_cv)

        last_state = facing
        last_switch = now

        # Resize to fit screen if needed
        screen_res = (1920, 1080)
        display_img_resized = cv2.resize(display_img, screen_res, interpolation=cv2.INTER_AREA)
        cv2.imshow("Display", display_img_resized)

    # Optional: show camera preview in a small window
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Camera", 20, 20)  # Position preview window at top-left
    cv2.imshow("Camera", preview_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
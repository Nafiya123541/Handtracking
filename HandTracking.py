import cv2
import mediapipe as mp
import time
from pathlib import Path

# MediaPipe Tasks API setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "hand_landmarker.task was not found. Put the model file in the same folder as this script."
    )

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

landmarker = HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

pTime = 0
last_timestamp = 0

while True:
    success, img = cap.read()

    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=imgRGB
    )

    timestamp = int(time.time() * 1000)
    timestamp = max(timestamp, last_timestamp + 1)
    last_timestamp = timestamp

    results = landmarker.detect_for_video(mp_image, timestamp)

    if results.hand_landmarks:
        h, w, c = img.shape

        for handLms in results.hand_landmarks:
            for id, lm in enumerate(handLms):
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Draw each hand landmark
                cv2.circle(img, (cx, cy), 6, (255, 0, 255), cv2.FILLED)

                # Highlight wrist landmark
                if id == 0:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            # Draw hand connections
            for start, end in connections:
                x1 = int(handLms[start].x * w)
                y1 = int(handLms[start].y * h)
                x2 = int(handLms[end].x * w)
                y2 = int(handLms[end].y * h)

                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime != pTime else 0
    pTime = cTime

    cv2.putText(
        img, str(int(fps)), (10, 70),
        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
    )

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()

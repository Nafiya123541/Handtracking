import cv2
import mediapipe as mp
import time
from pathlib import Path


class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        model_path = Path(__file__).with_name("hand_landmarker.task")

        if not model_path.exists():
            raise FileNotFoundError(
                "hand_landmarker.task was not found. Put the model file in the same folder as HandTrackingModule.py."
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=maxHands,
            min_hand_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon
        )

        self.landmarker = HandLandmarker.create_from_options(options)
        self.results = None
        self.last_timestamp = 0

        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=imgRGB
        )

        timestamp = int(time.time() * 1000)
        timestamp = max(timestamp, self.last_timestamp + 1)
        self.last_timestamp = timestamp

        self.results = self.landmarker.detect_for_video(mp_image, timestamp)

        if self.results.hand_landmarks and draw:
            h, w, c = img.shape

            for handLms in self.results.hand_landmarks:
                for lm in handLms:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 6, (255, 0, 255), cv2.FILLED)

                for start, end in self.connections:
                    x1 = int(handLms[start].x * w)
                    y1 = int(handLms[start].y * h)
                    x2 = int(handLms[end].x * w)
                    y2 = int(handLms[end].y * h)
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []

        if self.results and self.results.hand_landmarks:
            if handNo < len(self.results.hand_landmarks):
                myHand = self.results.hand_landmarks[handNo]
                h, w, c = img.shape

                for id, lm in enumerate(myHand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

                    if draw:
                        cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)

        return lmList

    def close(self):
        self.landmarker.close()


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()

        if not success:
            break

        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            print(lmList[4])

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
    detector.close()


if __name__ == "__main__":
    main()

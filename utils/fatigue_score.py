# fatigue_score.py

import cv2
import time
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist


# =========================================================
# MEDIAPIPE SETUP
# =========================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================================================
# LANDMARKS
# =========================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

UPPER_LIP = 13
LOWER_LIP = 14

NOSE = 1
CHIN = 152

LEFT_FACE = 234
RIGHT_FACE = 454


# =========================================================
# FATIGUE CLASS
# =========================================================

class FatigueScorer:

    def __init__(self):

        self.blink_count = 0
        self.yawn_count = 0

        self.prev_eye_closed = False
        self.prev_yawn = False

        self.eye_closed_start = None

        self.start_time = time.time()

        self.fatigue_score = 0

    # =====================================================
    # EAR
    # =====================================================

    def eye_aspect_ratio(self, eye):

        A = dist.euclidean(eye[1], eye[5])

        B = dist.euclidean(eye[2], eye[4])

        C = dist.euclidean(eye[0], eye[3])

        ear = (A + B) / (2.0 * C)

        return ear

    # =====================================================
    # NORMALIZE
    # =====================================================

    def normalize(self, value, min_val, max_val):

        value = max(min_val, min(value, max_val))

        return (value - min_val) / (max_val - min_val)

    # =====================================================
    # CALCULATE SCORE
    # =====================================================

    def calculate_score(
        self,
        eye_closure_duration,
        blink_rate,
        mouth_ratio,
        yawn_frequency,
        vertical_ratio,
        eye_slope,
        side_ratio
    ):

        E = self.normalize(eye_closure_duration, 0, 3) * 100

        B = self.normalize(blink_rate, 0, 40) * 100

        M = self.normalize(mouth_ratio, 0, 1) * 100

        Y = self.normalize(yawn_frequency, 0, 10) * 100

        V = self.normalize(vertical_ratio, 0, 1) * 100

        S = self.normalize(abs(eye_slope), 0, 15) * 100

        H = self.normalize(side_ratio, 0, 1) * 100

        current_score = (
            0.35 * E +
            0.20 * B +
            0.20 * M +
            0.10 * Y +
            0.10 * V +
            0.03 * S +
            0.02 * H
        )

        current_score = max(0, min(100, current_score))

        # Smooth

        self.fatigue_score = (
            0.8 * self.fatigue_score +
            0.2 * current_score
        )

        return round(self.fatigue_score, 2)

    # =====================================================
    # STATUS
    # =====================================================

    def get_status(self):

        if self.fatigue_score < 30:
            return "Normal"

        elif self.fatigue_score < 60:
            return "Tired"

        else:
            return "Drowsy"


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    fatigue_system = FatigueScorer()

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        h, w, _ = frame.shape

        if results.multi_face_landmarks:

            face_landmarks = results.multi_face_landmarks[0]

            landmarks = []

            for lm in face_landmarks.landmark:

                x = int(lm.x * w)

                y = int(lm.y * h)

                landmarks.append((x, y))

            # =====================================================
            # EYES
            # =====================================================

            left_eye = [landmarks[i] for i in LEFT_EYE]

            right_eye = [landmarks[i] for i in RIGHT_EYE]

            left_ear = fatigue_system.eye_aspect_ratio(left_eye)

            right_ear = fatigue_system.eye_aspect_ratio(right_eye)

            ear = (left_ear + right_ear) / 2.0

            # Draw eyes

            for point in left_eye + right_eye:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

            eyes_closed = ear < 0.22

            # Blink count

            if eyes_closed and not fatigue_system.prev_eye_closed:
                fatigue_system.blink_count += 1

            fatigue_system.prev_eye_closed = eyes_closed

            # Eye closure duration

            if eyes_closed:

                if fatigue_system.eye_closed_start is None:
                    fatigue_system.eye_closed_start = time.time()

                eye_closure_duration = (
                    time.time() - fatigue_system.eye_closed_start
                )

            else:

                fatigue_system.eye_closed_start = None

                eye_closure_duration = 0

            # =====================================================
            # YAWN
            # =====================================================

            upper_lip = landmarks[UPPER_LIP]

            lower_lip = landmarks[LOWER_LIP]

            mouth_distance = dist.euclidean(
                upper_lip,
                lower_lip
            )

            mouth_ratio = mouth_distance / 50

            yawning = mouth_ratio > 0.7

            if yawning and not fatigue_system.prev_yawn:
                fatigue_system.yawn_count += 1

            fatigue_system.prev_yawn = yawning

            # Draw lips

            cv2.circle(frame, upper_lip, 3, (255, 0, 0), -1)

            cv2.circle(frame, lower_lip, 3, (255, 0, 0), -1)

            # =====================================================
            # HEAD DOWN
            # =====================================================

            nose = landmarks[NOSE]

            chin = landmarks[CHIN]

            vertical_ratio = abs(chin[1] - nose[1]) / h

            # =====================================================
            # SIDE DISTRACTION
            # =====================================================

            left_face = landmarks[LEFT_FACE]

            right_face = landmarks[RIGHT_FACE]

            side_ratio = abs(
                (nose[0] - left_face[0]) -
                (right_face[0] - nose[0])
            ) / w

            # =====================================================
            # EYE SLOPE
            # =====================================================

            eye_slope = abs(
                left_eye[0][1] - right_eye[0][1]
            )

            # =====================================================
            # RATES
            # =====================================================

            elapsed_minutes = (
                time.time() - fatigue_system.start_time
            ) / 60

            if elapsed_minutes == 0:
                elapsed_minutes = 1

            blink_rate = (
                fatigue_system.blink_count /
                elapsed_minutes
            )

            yawn_frequency = (
                fatigue_system.yawn_count /
                elapsed_minutes
            )

            # =====================================================
            # FATIGUE SCORE
            # =====================================================

            fatigue = fatigue_system.calculate_score(
                eye_closure_duration,
                blink_rate,
                mouth_ratio,
                yawn_frequency,
                vertical_ratio,
                eye_slope,
                side_ratio
            )

            status = fatigue_system.get_status()

            # =====================================================
            # UI
            # =====================================================

            cv2.putText(
                frame,
                f"Fatigue Level: {fatigue:.1f}%",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"Status: {status}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Blinks: {fatigue_system.blink_count}",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Yawns: {fatigue_system.yawn_count}",
                (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (20, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # Drowsy alert

            if fatigue > 65:

                cv2.putText(
                    frame,
                    "DROWSINESS ALERT!",
                    (250, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

        cv2.imshow("Fatigue Detection", frame)

        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()
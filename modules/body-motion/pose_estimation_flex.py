import cv2
import mediapipe as mp
import numpy as np
from angle_flex import calculate_angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# Globali per contatori e stadi dei movimenti
counters = {
    "Flex left arm": 0,
    "Flex right arm": 0,
    "Rise arms": 0
}

stages = {
    "Flex left arm": None,
    "Flex right arm": None,
    "Rise arms": None
}

def process_pose_frame(frame):
    global counters, stages

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    results = pose_detector.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        landmarks = results.pose_landmarks.landmark
        h, w = frame.shape[:2]

        # --- FLEX LEFT ARM ---
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
        cv2.putText(image, f"{int(angle_left)}", tuple(np.multiply(left_elbow,[w,h]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

        if angle_left > 160:
            stages["Flex left arm"] = "down"
        if angle_left < 30 and stages["Flex left arm"] == "down":
            stages["Flex left arm"] = "up"
            counters["Flex left arm"] += 1

        # --- FLEX RIGHT ARM ---
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        angle_right = calculate_angle(right_shoulder, right_elbow, right_wrist)
        cv2.putText(image, f"{int(angle_right)}", tuple(np.multiply(right_elbow,[w,h]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

        if angle_right > 160:
            stages["Flex right arm"] = "down"
        if angle_right < 30 and stages["Flex right arm"] == "down":
            stages["Flex right arm"] = "up"
            counters["Flex right arm"] += 1

        # --- RISE ARMS (es. braccia sopra la testa) ---
        # Usa le spalle e i polsi per rilevare se le braccia sono alzate
        left_y = left_wrist[1]
        right_y = right_wrist[1]
        shoulder_y = left_shoulder[1]  # approssimazione

        if left_y < shoulder_y and right_y < shoulder_y:
            if stages["Rise arms"] != "up":
                stages["Rise arms"] = "up"
                counters["Rise arms"] += 1
        else:
            stages["Rise arms"] = "down"

    except:
        pass

    # Disegna landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

    return image, counters

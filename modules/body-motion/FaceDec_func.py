import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

previous_pose = "Forward"
counters = {
    "Looking Left": 0,
    "Looking Right": 0,
    "Looking Down": 0,
    "Looking Up": 0,
    "Forward": 0
}

def process_frame(frame):
    global previous_pose, counters

    if frame is None:
        return frame, counters

    height, width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)

    frame_out = frame.copy()

    if result.multi_face_landmarks:
        face_3d, face_2d = [], []

        for face_landmarks in result.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                #cv2.circle(mask, (cx, cy), 2, (0, 255, 0), -1)  # Disegna cerchi verdi per la maschera
                cv2.circle(frame_out, (cx, cy), 2, (0, 255, 0), -1)  # Punti verdi direttamente sul video

        # Applicare la maschera sul frame
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # face_only = cv2.bitwise_and(frame, mask)

        # # Cambiare il colore della maschera
        # new_color = (0, 255, 0)
        # non_black_pixels = (face_only != [0, 0, 0]).all(axis=2)
        # face_only[non_black_pixels] = new_color

        # Extract key landmarks for head pose
                
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:
                    x, y = int(lm.x * width), int(lm.y * height)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = width
        cam_matrix = np.array([[focal_length, 0, height / 2],
                               [0, focal_length, width / 2],
                               [0, 0, 1]], dtype=np.float64)
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        x_angle = angles[0] * 360
        y_angle = angles[1] * 360

        # Determina la direzione
        if y_angle < -10:
            head_direction = "Looking Left"
        elif y_angle > 10:
            head_direction = "Looking Right"
        elif x_angle < -10:
            head_direction = "Looking Down"
        elif x_angle > 10:
            head_direction = "Looking Up"
        else:
            head_direction = "Forward"

        # Aggiorna i contatori solo se cambia direzione
        if head_direction != previous_pose:
            counters[head_direction] += 1
            previous_pose = head_direction

        return frame_out, counters

    return frame, counters

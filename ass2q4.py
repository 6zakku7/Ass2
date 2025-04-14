import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

counter = 0
stage = None
prev_nose_y = 0
nod_threshold = 0.03

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        landmarks = results.pose_landmarks.landmark

        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        nose_y = nose.y

        # Nod detection logic
        if prev_nose_y == 0:
            prev_nose_y = nose_y

        # Calculate vertical movement
        vertical_movement = nose_y - prev_nose_y

        # Down phase
        if vertical_movement > nod_threshold and stage != 'down':
            stage = 'down'

        # Up phase (count as rep when coming back up)
        if vertical_movement < -nod_threshold and stage == 'down':
            stage = 'up'
            counter += 1

        prev_nose_y = nose_y

    except:
        pass

    # Setup status box
    cv2.rectangle(image, (0, 0), (300, 120), (245, 117, 16), -1)

    # Nod counter display
    cv2.putText(image, 'HEAD NOD COUNTER', (15, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, 'REPS', (15, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, str(counter), (15, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Stage display
    cv2.putText(image, 'STAGE', (320, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, stage if stage else 'waiting...', (320, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    #head position
    indicator_y = int(nose.y * image.shape[0])
    cv2.circle(image, (50, indicator_y), 15, (0, 255, 0), -1)
    cv2.line(image, (30, int(image.shape[0] * 0.3)), (70, int(image.shape[0] * 0.3)), (255, 255, 255),
             2)  # Upper threshold
    cv2.line(image, (30, int(image.shape[0] * 0.7)), (70, int(image.shape[0] * 0.7)), (255, 255, 255),
             2)  # Lower threshold

    # Render detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    cv2.imshow('Head Nod Counter', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
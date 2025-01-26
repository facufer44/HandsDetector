import cv2

import mediapipe as mp

#This line if to capture image with the webcam
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands

hands = mpHands.Hands(static_image_mode = False, 
                      max_num_hands = 2, 
                      min_detection_confidence = 0.9, 
                      min_tracking_confidence = 0.8)

mpDraw = mp.solutions.drawing_utils

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el video")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result= hands.process(rgb_frame)


#Code for drawing and detecting the fingers
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(
                frame, hand_landmarks,
                mpHands.HAND_CONNECTIONS
            )

#Code for detecting the number of hands on the screen
    if result.multi_hand_landmarks:
        num_hands = len(result.multi_hand_landmarks)
        cv2.putText(frame, f'Hands detected: {num_hands}', 
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)


#Code for calculate the distance between two fingers
    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            hand_label = result.multi_handedness[idx].classification[0].label
            label_position = (int(hand_landmarks.landmark[0].x * frame.shape[1]),
                          int(hand_landmarks.landmark[0].y * frame.shape[0]))
        cv2.putText(frame, hand_label, label_position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# this part is for detecting if the hand is closed
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            fingers_folded = True
        for tip_id in [8, 12, 16, 20]:  # Puntas de los dedos excepto el pulgar
            if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                fingers_folded = False
                break
        if fingers_folded:
            cv2.putText(frame, 'Hand closed', (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


    cv2.imshow('Deteccion de manos', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()


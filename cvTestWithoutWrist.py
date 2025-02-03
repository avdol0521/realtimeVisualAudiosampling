import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Flip for mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB

    results = hands.process(rgb_frame)  # Detect hands

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the positions of the index finger tip (8) and thumb tip (4)
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            # Convert normalized coordinates to pixel values
            h, w, _ = frame.shape
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Draw circles on fingertips
            cv2.circle(frame, (index_x, index_y), 10, (255, 255, 255), -1)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 255, 255), -1)
            cv2.circle(frame, (index_x, index_y), 5, (255, 0, 0), -1)
            cv2.circle(frame, (thumb_x, thumb_y), 5, (255, 0, 0), -1)

            # Draw line between index and thumb
            cv2.line(frame, (index_x, index_y), (thumb_x, thumb_y), (0, 255, 0), 5)

    # Show the frame
    cv2.imshow("Finger Tracking", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
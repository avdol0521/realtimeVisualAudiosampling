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
        hands_info = []  # Store hand info (x position and label)

        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get the x-coordinate of the wrist (landmark 0)
            wrist_x = hand_landmarks.landmark[0].x

            # Assume first detected hand is left, then compare
            hand_label = "Left" if len(hands_info) == 0 else "Right"
            hands_info.append((wrist_x, hand_label, hand_landmarks))

        # Sort hands based on wrist_x (leftmost hand is "Left")
        hands_info.sort(key=lambda x: x[0])  
        if len(hands_info) == 2:
            hands_info[0] = (hands_info[0][0], "Left", hands_info[0][2])
            hands_info[1] = (hands_info[1][0], "Right", hands_info[1][2])

        # Draw labels and finger lines
        for wrist_x, hand_label, hand_landmarks in hands_info:
            h, w, _ = frame.shape

            # Get fingertip positions
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Draw circles on fingertips
            cv2.circle(frame, (index_x, index_y), 10, (255, 255, 255), -1)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 255, 255), -1)

            # Draw line between index and thumb
            cv2.line(frame, (index_x, index_y), (thumb_x, thumb_y), (0, 255, 0), 5)

            # Display hand label
            cv2.putText(frame, hand_label, (index_x - 20, index_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Skibidi", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

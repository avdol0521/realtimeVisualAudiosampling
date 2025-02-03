import cv2  # OpenCV for handling the webcam
import mediapipe as mp  # MediaPipe for hand tracking

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils  # Utility to draw hand landmarks

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()  # Capture a frame
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Flip horizontally (like a mirror)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB (MediaPipe requirement)
    
    # Process the frame for hand detection
    results = hands.process(rgb_frame)

    # If hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # Draw hand landmarks

    # Display the frame
    cv2.imshow("Hand Tracking", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

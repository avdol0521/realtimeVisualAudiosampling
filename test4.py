import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import soundfile as sf

# Load audio file
audio_file = "J.wav"
audio_data, sample_rate = sf.read(audio_file)  # Read audio file
audio_playback = None  # Placeholder for current playback process

# Convert stereo to mono if needed
if len(audio_data.shape) > 1:
    audio_data = np.mean(audio_data, axis=1)
    
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(np.array(p1) - np.array(p2))

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Flip for mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB

    results = hands.process(rgb_frame)  # Detect hands

    hands_info = []  # Store hand info (x position and label)
    hand_lines = {}  # Store line lengths

    if results.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get the wrist position to determine left/right
            wrist_x = hand_landmarks.landmark[0].x
            hand_label = "Left" if len(hands_info) == 0 else "Right"
            hands_info.append((wrist_x, hand_label, hand_landmarks))

        # Ensure the leftmost hand is always "Left"
        hands_info.sort(key=lambda x: x[0])
        if len(hands_info) == 2:
            hands_info[0] = (hands_info[0][0], "Left", hands_info[0][2])
            hands_info[1] = (hands_info[1][0], "Right", hands_info[1][2])

        for wrist_x, hand_label, hand_landmarks in hands_info:
            h, w, _ = frame.shape

            # Get fingertip positions
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Draw circles and line
            cv2.circle(frame, (index_x, index_y), 10, (255, 255, 255), -1)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 255, 255), -1)
            cv2.line(frame, (index_x, index_y), (thumb_x, thumb_y), (0, 255, 0), 5)

            # Display hand label
            cv2.putText(frame, hand_label, (index_x - 20, index_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Calculate line length (normalized)
            hand_lines[hand_label] = calculate_distance((index_x, index_y), (thumb_x, thumb_y)) / w

    # Adjust audio parameters if both hands are detected
    if "Left" in hand_lines and "Right" in hand_lines:
        left_line = hand_lines["Left"]
        right_line = hand_lines["Right"]

        # Speed = Distance between both hand lines
        if len(hands_info) == 2:
            left_x = hands_info[0][2].landmark[8].x * w
            right_x = hands_info[1][2].landmark[8].x * w
            hand_distance = abs(right_x - left_x) / w  # Normalize distance

            playback_speed = np.clip(0.5 + hand_distance * 2, 0.5, 2.0)  # Speed range: 0.5x to 2x
        else:
            playback_speed = 1.0

        # Pitch = Left-hand line length
        pitch_factor = np.clip(1.0 + (left_line - 0.1) * 2, 0.5, 2.0)  # Scale pitch

        # Volume = Right-hand line length
        volume = np.clip(right_line * 2, 0.1, 1.0)  # Scale volume

        # Adjust audio playback
        if audio_playback is None or playback_speed != 1.0:
            if audio_playback is not None:
                audio_playback.stop()

            # Adjust audio speed
            new_sample_rate = int(sample_rate * playback_speed)
            new_audio = np.interp(np.linspace(0, len(audio_data), num=int(len(audio_data) * playback_speed)),
                                np.arange(len(audio_data)), audio_data)

            # Play adjusted audio
            audio_playback = sd.play(new_audio * volume, samplerate=new_sample_rate)

    # Show the frame
    cv2.imshow("Hand Tracking", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sd.stop()

import cv2 
import mediapipe as mp
import math 


def circle_edge(deg, cx, cy, r):
    points = []
    for i in range(deg):
        radian = math.radians(i)
        x = int(cx + (r * math.cos(radian)))
        y = int(cy + (r * math.sin(radian)))
        points.append((x, y))
    return points

def drawSkibidi(array):
    for i in range(len(array)):
        if i == len(array) - 1:
            cv2.line(frame, array[i], array[0], color, thickness)
        else:
            cv2.line(frame, array[i], array[i+1], color, thickness) 

def calcDistance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

mp_hands = mp.solutions.hands # Load the MediaPipe Hands model
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) # Set confidence thresholds
mp_draw = mp.solutions.drawing_utils # Load the drawing utility

cap = cv2.VideoCapture(0) # Open the webcam

while True:
    success, frame = cap.read() # Read a frame
    if not success:
        break # Break the loop if reading was unsuccessful
    frame = cv2.flip(frame, 1) # Flip the frame horizontally
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB
    
    hand_frames = hands.process(rgb_frame) # Process the frame for hand detection
    if hand_frames.multi_hand_landmarks: # If hands are detected
        hands_info = []
        speed_hand = None
        frequency_hand = None
        for hand_index, hand_landmarks in enumerate(hand_frames.multi_hand_landmarks):
            wrist_x = hand_landmarks.landmark[0].x
            hand_label = "speed" if len(hands_info) == 0 else "freaquency" # Assume first detected hand is left, then compare
            hands_info.append((wrist_x, hand_label, hand_landmarks))
        hands_info.sort(key=lambda x: x[0]) # Sort hands based on wrist_x (leftmost hand is "Left")
        if len(hands_info) == 2:
            hands_info[0] = (hands_info[0][0], "speed", hands_info[0][2])
            hands_info[1] = (hands_info[1][0], "freaquency", hands_info[1][2])
            # speed_hand = hands_info[0][2]
            # frequency_hand = hands_info[1][2]
        
        for wrist_x, hand_label, hand_landmarks in hands_info:
            index_tip = hand_landmarks.landmark[8] # Get the position of the index fingertip
            thumb_tip = hand_landmarks.landmark[4] # Get the position of the thumb fingertip
            h, w, _ = frame.shape # h = 480, w = 640 
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h) # Convert normalized coordinates to pixel values
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h) # Convert normalized coordinates to pixel values
            degrees = 360 
            radius = 10 
            thickness = 1
            color = (255, 255, 255) 
            circle_points1 = circle_edge(degrees, index_x, index_y, radius)
            circle_points2 = circle_edge(degrees, thumb_x, thumb_y, radius)
            # index_tip_speed = speed_hand.landmark[8]
            # thumb_tip_speed = speed_hand.landmark[4]
            # index_x_speed, index_y_speed = int(index_tip_speed.x * w), int(index_tip_speed.y * h)
            # thumb_x_speed, thumb_y_speed = int(thumb_tip_speed.x * w), int(thumb_tip_speed.y * h)
            # speed_distance = calcDistance(index_x_speed, index_y_speed, thumb_x_speed, thumb_y_speed)
            # print("speed_distance")
            # print(speed_distance)
            # index_tip_frequency = frequency_hand.landmark[8]
            # thumb_tip_frequency = frequency_hand.landmark[4]
            # index_x_frequency, index_y_frequency = int(index_tip_frequency.x * w), int(index_tip_frequency.y * h)
            # thumb_x_frequency, thumb_y_frequency = int(thumb_tip_frequency.x * w), int(thumb_tip_frequency.y * h)
            # frequency_distance = calcDistance(index_x_frequency, index_y_frequency, thumb_x_frequency, thumb_y_frequency)
            # print("frequency_distance")
            # print(frequency_distance)
            drawSkibidi(circle_points1)
            drawSkibidi(circle_points2)
            cv2.line(frame, (index_x, index_y), (thumb_x, thumb_y), color, thickness)
            cv2.putText(frame, hand_label, (index_x - 20, index_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)

    cv2.imshow("skibidi", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
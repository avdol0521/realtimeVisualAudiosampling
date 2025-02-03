import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
from collections import deque
from scipy.signal import resample
import time

# Load the audio file
audio_file = "J.wav"
audio_data, sample_rate = sf.read(audio_file)

# Convert stereo to mono if needed
if len(audio_data.shape) > 1:
    audio_data = np.mean(audio_data, axis=1)

# Audio processing parameters
BUFFER_SIZE = 2048
CHUNK_SIZE = 4096
audio_buffer = deque(maxlen=CHUNK_SIZE*2)

# Hand tracking setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Audio control variables
playback_speed = 1.0
frequency_factor = 1.0
volume = 1.0
read_position = 0
lock = threading.Lock()

def process_audio_chunk(chunk, frequency_factor):
    if frequency_factor == 1.0:
        return chunk  # No change if frequency is 1.0
    
    chunk_length = len(chunk)
    resampled_length = int(chunk_length * frequency_factor)  # Adjust frequency
    
    if resampled_length <= 0:
        return np.zeros(1)
    
    return resample(chunk, resampled_length)

def audio_callback(outdata, frames, time, status):
    global read_position, audio_buffer
    
    if status:
        print(status)
    
    with lock:
        if len(audio_buffer) < frames:
            chunk = audio_data[read_position:read_position + CHUNK_SIZE]
            if len(chunk) == 0:
                read_position = 0
                chunk = audio_data[read_position:read_position + CHUNK_SIZE]
            
            processed_chunk = process_audio_chunk(chunk, frequency_factor)
            audio_buffer.extend(processed_chunk)
            read_position = (read_position + CHUNK_SIZE) % len(audio_data)
        
        output_samples = np.array([audio_buffer.popleft() for _ in range(min(frames, len(audio_buffer)))])
        
        if len(output_samples) < frames:
            output_samples = np.pad(output_samples, (0, frames - len(output_samples)))
        
        outdata[:] = (output_samples * volume).reshape(-1, 1)

# Start audio stream
stream = sd.OutputStream(
    samplerate=sample_rate,
    channels=1,
    callback=audio_callback,
    blocksize=BUFFER_SIZE
)
stream.start()

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    
    results = hands.process(frame_rgb)
    frame_rgb.flags.writeable = True
    
    left_hand = right_hand = None
    left_line_length = right_line_length = 0
    
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            
            index_pos = (int(index_tip.x * w), int(index_tip.y * h))
            thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            
            # Calculate line length
            line_length = np.linalg.norm(np.array(index_pos) - np.array(thumb_pos))
            
            if label == "Left":
                left_hand = (index_pos, thumb_pos)
                left_line_length = line_length
            else:
                right_hand = (index_pos, thumb_pos)
                right_line_length = line_length
            
            # Draw markers and lines
            cv2.circle(frame, index_pos, 8, (255, 255, 255), -1)
            cv2.circle(frame, thumb_pos, 8, (255, 255, 255), -1)
            cv2.line(frame, index_pos, thumb_pos, (0, 255, 0), 2)
    
    if left_hand and right_hand:
        # Calculate distance between the two lines
        left_center = np.mean([left_hand[0], left_hand[1]], axis=0)
        right_center = np.mean([right_hand[0], right_hand[1]], axis=0)
        lines_distance = np.linalg.norm(left_center - right_center)
        
        max_distance = w * 0.8  # Maximum expected distance between lines
        max_line_length = 150   # Maximum expected line length
        
        with lock:
            # Left line length controls playback speed
            playback_speed = np.clip(left_line_length / max_line_length, 0.5, 2.0)
            
            # Distance between hands controls volume
            volume = np.clip(lines_distance / max_distance, 0.2, 1.0)
            
            # Right hand controls frequency shift
            frequency_factor = np.clip(right_line_length / max_line_length, 0.5, 2.0)
    
    cv2.imshow("Hand Audio Controller", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
stream.stop()
stream.close()
import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
from collections import deque
import time

# Load the audio file
audio_file = "J.wav"
audio_data, sample_rate = sf.read(audio_file)

# Convert stereo to mono if needed
if len(audio_data.shape) > 1:
    audio_data = np.mean(audio_data, axis=1)

# Audio processing parameters
BUFFER_SIZE = 2048
CHUNK_SIZE = 4096  # Size of chunks to pre-process
audio_buffer = deque(maxlen=CHUNK_SIZE*2)

# Hand tracking setup with reduced CPU usage
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3
)

# Audio control variables
playback_speed = 1.0
pitch_factor = 1.0
volume = 1.0
read_position = 0
processing_time = 0
lock = threading.Lock()

def process_audio_chunk(chunk, pitch):
    """Process a small chunk of audio instead of the entire file"""
    if pitch == 1.0:
        return chunk
    
    chunk_length = len(chunk)
    resampled_length = int(chunk_length * pitch)
    
    if resampled_length <= 0:
        return np.zeros(1)
        
    indices = np.linspace(0, chunk_length-1, resampled_length)
    return np.interp(indices, np.arange(chunk_length), chunk)

def audio_callback(outdata, frames, time, status):
    global read_position, audio_buffer
    
    if status:
        print(status)
    
    with lock:
        if len(audio_buffer) < frames:
            # Refill buffer if running low
            chunk = audio_data[read_position:read_position + CHUNK_SIZE]
            if len(chunk) == 0:
                read_position = 0
                chunk = audio_data[read_position:read_position + CHUNK_SIZE]
            
            processed_chunk = process_audio_chunk(chunk, pitch_factor)
            audio_buffer.extend(processed_chunk)
            read_position = (read_position + CHUNK_SIZE) % len(audio_data)
        
        # Get required samples from buffer
        output_samples = np.array([audio_buffer.popleft() for _ in range(min(frames, len(audio_buffer)))])
        
        # Apply volume and reshape
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

# Initialize video capture with lower resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Performance monitoring
frame_time = time.time()
fps_update_interval = 1.0  # Update FPS every second
fps = 0
frames_counted = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Calculate and display FPS
    current_time = time.time()
    frames_counted += 1
    if current_time - frame_time >= fps_update_interval:
        fps = frames_counted / (current_time - frame_time)
        frames_counted = 0
        frame_time = current_time
    
    # Flip and process frame
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Convert to RGB without copying the frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False  # Prevent copy in MediaPipe
    
    # Process hands
    results = hands.process(frame_rgb)
    frame_rgb.flags.writeable = True
    
    left_hand = right_hand = None
    
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            
            # Get finger positions
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            
            # Convert to pixel coordinates
            index_pos = (int(index_tip.x * w), int(index_tip.y * h))
            thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            
            if label == "Left":
                left_hand = (index_pos, thumb_pos)
            else:
                right_hand = (index_pos, thumb_pos)
            
            # Draw markers
            cv2.circle(frame, index_pos, 8, (255, 255, 255), -1)
            cv2.circle(frame, thumb_pos, 8, (255, 255, 255), -1)
            cv2.line(frame, index_pos, thumb_pos, (0, 255, 0), 2)
    
    # Update audio parameters
    if left_hand and right_hand:
        left_dist = np.linalg.norm(np.array(left_hand[0]) - np.array(left_hand[1]))
        right_dist = np.linalg.norm(np.array(right_hand[0]) - np.array(right_hand[1]))
        hand_gap = np.linalg.norm(np.array(left_hand[0]) - np.array(right_hand[0]))
        
        max_hand_gap = w / 2
        max_finger_dist = 150  # Reduced for better control
        
        with lock:
            playback_speed = np.clip(hand_gap / max_hand_gap, 0.5, 2.0)
            pitch_factor = np.clip(left_dist / max_finger_dist, 0.5, 2.0)
            volume = np.clip(right_dist / max_finger_dist, 0.2, 1.0)
    
    # Display FPS and controls
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Speed: {playback_speed:.2f}x", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Pitch: {pitch_factor:.2f}x", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Volume: {volume:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Hand Audio Controller", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
stream.stop()
stream.close()
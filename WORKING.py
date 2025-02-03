import cv2 
import mediapipe as mp
import math 
import numpy as np
import pyaudio
import wave
import threading 
from queue import Queue
from scipy.signal import resample

class Playskibidi: 
    def __init__(self, filename):
        self.wf = wave.open(filename, 'rb')
        self.p = pyaudio.PyAudio()

        self.channels = self.wf.getnchannels()
        self.sample_width = self.wf.getsampwidth()
        self.sample_rate = self.wf.getframerate()

        self._volume = 1.0
        self._speed = 1.0
        self._frequency = 1.0
        self._playing = False
        self._lock = threading.Lock()

        self.chunk_size = 1024 
        self.audio_queue = Queue(maxsize=20)

        self.stream = self.p.open(format=self.p.get_format_from_width(self.sample_width),channels=self.channels,rate=self.sample_rate,output=True,stream_callback=self._callback)
    def _callback(self, in_data, frame_count, time_info, status):
        
        data = self.wf.readframes(frame_count)
        if not data:
            self.wf.rewind()
            data = self.wf.readframes(frame_count) 
        audio_data = np.frombuffer(data, dtype=np.int16) # convert to numpy array for processing 

        with self._lock:
            audio_data = audio_data * self._volume # apply volume
            
            if self._frequency != 1.0: # frequency shift using interpolation
                x = np.linspace(0, len(audio_data), len(audio_data))
                x_new = np.linspace(0, len(audio_data), int(len(audio_data) * self._frequency))
                audio_data = np.interp(x_new, x, audio_data)
            
            if self._speed != 1.0: # apply speed by resampling the audio data
                chunk_length = len(audio_data)
                time_steps = np.arange(chunk_length)
                # resampled_steps = np.linspace(0, chunk_length, int(chunk_length / self._speed))
                resampled_steps = int(chunk_length / self._speed)
                audio_data = resample(audio_data, resampled_steps)
            
            # this fucker caused me so much pain fuck this bitch
            skibidi_final_audio = audio_data.astype(np.int16)
            # print("final audio: ", skibidi_final_audio)
            desired_length = frame_count * self.channels
            # print("desired length: ", desired_length)
            if len(skibidi_final_audio) < desired_length:
                skibidi_final_audio = np.pad(skibidi_final_audio, (0, desired_length - len(skibidi_final_audio)), mode='constant')
            elif len(skibidi_final_audio) > desired_length:
                skibidi_final_audio = skibidi_final_audio[:desired_length]
        return (skibidi_final_audio.tobytes(), pyaudio.paContinue)
    
    def play(self):
        with self._lock:
            self._playing = True
            self.stream.start_stream()
    def pause(self):
        with self._lock:
            self._playing = False
    def set_volume(self, volume):
        with self._lock:
            self._volume = max(0.1, min(2.0, volume))
    def set_speed(self, speed):
        with self._lock:
            self._speed = max(0.1, min(2.0, speed))
    def set_frequency(self, frequency):
        with self._lock:
            self._frequency = max(0.1, min(2.0, frequency))
    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.wf.close()

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

def calcAvg(val1, val2):
    return ((val1 + val2) / 2)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

min_val, max_val = 0.1, 2.0 
fuckyou = 300
fuckyoutwo = 250
degrees = 360 
radius = 10 
radius2 = 12
thickness = 1
color = (255, 255, 255)

player = Playskibidi("J.wav") # INPUT WAV FILE PATH/NAME HERE
speed_value = 1.0
frequency_value = 1.0
volume_value = 1.0

player.play()

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_frames = hands.process(rgb_frame)
    if hand_frames.multi_hand_landmarks:
        hands_info = []
        speed_hand = None
        frequency_hand = None
        for hand_index, hand_landmarks in enumerate(hand_frames.multi_hand_landmarks):
            wrist_x = hand_landmarks.landmark[0].x
            hand_label = "speed" if len(hands_info) == 0 else "freaquency"
            hands_info.append((wrist_x, hand_label, hand_landmarks))

        hands_info.sort(key=lambda x: x[0])
        # if len(hands_info) == 2:
        #     hands_info[0] = (hands_info[0][0], "speed", hands_info[0][2])
        #     hands_info[1] = (hands_info[1][0], "freaquency", hands_info[1][2])
        #     speed_hand = hands_info[0][2]
        #     frequency_hand = hands_info[1][2]
        if len(hands_info) == 1:
            # Duplicate the single hand for both roles.
            hands_info[0] = (hands_info[0][0], "2 hands pls", hands_info[0][2])
            speed_hand = hands_info[0][2]
            frequency_hand = hands_info[0][2]
        elif len(hands_info) >= 2:
            hands_info[0] = (hands_info[0][0], "pitch maybe???", hands_info[0][2])
            hands_info[1] = (hands_info[1][0], "freaquency???", hands_info[1][2])
            speed_hand = hands_info[0][2]
            frequency_hand = hands_info[1][2]

        for wrist_x, hand_label, hand_landmarks in hands_info:
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            h, w, _ = frame.shape
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            circle_points1 = circle_edge(degrees, index_x, index_y, radius)
            circle_points2 = circle_edge(degrees, thumb_x, thumb_y, radius)
            # index_x_frequency, index_x_speed = 
            # thumb_x_frequency, thumb_x_speed =
            # index_y_frequency, index_y_speed = 
            # thumb_y_frequency, thumb_y_speed =
            if speed_hand:  # Prevents NoneType error
                index_tip_speed = speed_hand.landmark[8]
                thumb_tip_speed = speed_hand.landmark[4]
                index_x_speed, index_y_speed = int(index_tip_speed.x * w), int(index_tip_speed.y * h)
                thumb_x_speed, thumb_y_speed = int(thumb_tip_speed.x * w), int(thumb_tip_speed.y * h)
                speed_distance = calcDistance(index_x_speed, index_y_speed, thumb_x_speed, thumb_y_speed)
                speed_value = round((min_val + (max_val - min_val) * min(max(speed_distance / fuckyou, 0), 1)), 1)
                player.set_speed(speed_value)
                print("idk_value:", speed_value)
                print("idk_distance: ", speed_distance)

            if frequency_hand:  # Prevents NoneType error
                index_tip_frequency = frequency_hand.landmark[8]
                thumb_tip_frequency = frequency_hand.landmark[4]
                index_x_frequency, index_y_frequency = int(index_tip_frequency.x * w), int(index_tip_frequency.y * h)
                thumb_x_frequency, thumb_y_frequency = int(thumb_tip_frequency.x * w), int(thumb_tip_frequency.y * h)
                frequency_distance = calcDistance(index_x_frequency, index_y_frequency, thumb_x_frequency, thumb_y_frequency)
                frequency_value = round((min_val + (max_val - min_val) * min(max(frequency_distance / fuckyou, 0), 1)), 1)
                player.set_frequency(frequency_value)
                print("frequency_value:", frequency_value)
                print("frequency_distance: ", frequency_distance)

            if len(hands_info) == 2:
                volume_distance = calcAvg(calcDistance(index_x_speed, index_y_speed, index_x_frequency, index_y_frequency), calcDistance(thumb_x_speed, thumb_y_speed, thumb_x_frequency, thumb_y_frequency))
                volume_value = round((min_val + (max_val - min_val) * min(max(volume_distance / fuckyoutwo, 0), 1)), 1)
                print("volume markiplier: ", volume_value)
                player.set_volume(volume_value)
                print(" /\_/\ ")
                print("( o.o )")
                print(" > ^ < ")

                player.set_frequency(frequency_value)
                player.set_speed(speed_value)
            drawSkibidi(circle_points1)
            # drawSkibidi(circle_edge(degrees, index_x, index_y, radius2))
            drawSkibidi(circle_points2)
            # drawSkibidi(circle_edge(degrees, thumb_x, thumb_y, radius2))

            if len(hands_info) >= 1:  # Ensure at least one hand is detected
                player.set_frequency(frequency_value)
                player.set_speed(speed_value)

            if not player._playing:
                player.play()  # Restart playing if it has stopped
            cv2.line(frame, (index_x, index_y), (thumb_x, thumb_y), color, thickness)
            cv2.putText(frame, hand_label, (index_x - 20, index_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)
    cv2.imshow("skibidi", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        player.close()
        break

cap.release()
cv2.destroyAllWindows()
# .     .       .  .   . .   .   . .    +  .
#   .     .  :     .    .. :. .___---------___.
#        .  .   .    .  :.:. _".^ .^ ^.  '.. :"-_. .
#     .  :       .  .  .:../:            . .^  :.:\.
#         .   . :: +. :.:/: .   .    .        . . .:\
#  .  :    .     . _ :::/:               .  ^ .  . .:\
#   .. . .   . - : :.:./.                        .  .:\
#   .      .     . :..|:                    .  .  ^. .:|
#     .       . : : ..||        .                . . !:|
#   .     . . . ::. ::\(                           . :)/
#  .   .     : . : .:.|. ######              .#######::|
#   :.. .  :-  : .:  ::|.#######           ..########:|
#  .  .  .  ..  .  .. :\ ########          :######## :/
#   .        .+ :: : -.:\ ########       . ########.:/
#     .  .+   . . . . :.:\. #######       #######..:/
#       :: . . . . ::.:..:.\           .   .   ..:/
#    .   .   .  .. :  -::::.\.       | |     . .:/
#       .  :  .  .  .-:.":.::.\             ..:/
#  .      -.   . . . .: .:::.:.\.           .:/
# .   .   .  :      : ....::_:..:\   ___.  :/
#    .   .  .   .:. .. .  .: :.:.:\       :/
#      +   .   .   : . ::. :.:. .:.|\  .:/|
#      .         +   .  .  ...:: ..|  --.:|
# .      . . .   .  .  . ... :..:.."(  ..)"
#  .   .       .      :  .   .: ::/  .  .::\
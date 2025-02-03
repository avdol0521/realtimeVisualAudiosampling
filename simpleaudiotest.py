# import sounddevice as sd
# import numpy as np
# import wave

# # Load WAV file
# filename = "J.wav"
# wf = wave.open(filename, 'rb')

# # Read audio data
# data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

# # Play audio
# sd.play(data, samplerate=wf.getframerate())
# sd.wait()
import sounddevice as sd  # Library for audio playback
import numpy as np  # Used to handle audio data as arrays
import wave  # Used to read WAV files

# Open the WAV file for reading
filename = "J.wav"  # Your audio file
wf = wave.open(filename, 'rb')  # 'rb' means read in binary mode

# Extract important audio properties
samplerate = wf.getframerate()  # Sample rate (how many samples per second)
channels = wf.getnchannels()  # Number of audio channels (1 = mono, 2 = stereo)

# Read the raw audio data as bytes and convert it into a NumPy array of 16-bit integers
data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

# If the audio has 2 channels (stereo), reshape the array to separate left and right channels
if channels == 2:
    data = data.reshape(-1, 2)  # Reshape to a 2D array with two columns (left & right audio)

# Play the audio using the correct sample rate
sd.play(data, samplerate=samplerate)

# Wait until the audio playback is finished before continuing
sd.wait()

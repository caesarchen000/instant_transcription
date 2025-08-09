import os
import wave
import time
import threading
import tkinter as tk
import pyaudio
import sounddevice as sd
import soundfile as sf
import mss
import cv2
import numpy as np

class VoiceRecorder:

    def __init__(self):
        self.root = tk.Tk()
        self.root.resizable(False, False)
        self.button=tk.Button(text="Record",
                              font=("Arial",120,"bold"),
                              command=self.click_handler)
        self.button.pack()
        self.label=tk.Label(text="00:00:00",
                            font=("Arial",40,"bold"))
        self.label.pack()
        self.recording=False

        self.root.mainloop()
    
    def click_handler(self):
        if self.recording:
            self.recording=False
            self.button.config(fg="black")
            self.label.config(text="00:00:00")
        else:
            self.recording=True
            self.button.config(fg="red")
            print("Button clicked")
            threading.Thread(target=self.record_screen_audio).start()

    def record(self):
        audio=pyaudio.PyAudio()
        stream=audio.open(format=pyaudio.paInt16,
                          channels=1,
                          rate=44100,
                          input=True,
                          frames_per_buffer=1024)
        frames=[]
        start=time.time()
        while self.recording:
            data=stream.read(1024)
            frames.append(data)

            passed=time.time()-start
            seconds=int(passed%60)
            minutes=int(passed//60)
            hours=int(passed//3600)
            self.label.config(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
       
        exists=True
        i=1
        while exists:
            if os.path.exists(f"recordings/recording_{i}.wav"):
                i+=1
            else:
                exists=False
        sound_file=wave.open(f"recordings/recording_{i}.wav","wb")
        sound_file.setnchannels(1)
        sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(44100)
        sound_file.writeframes(b"".join(frames))
        sound_file.close()
        self.button.config(fg="black")
        self.label.config(text="00:00")
        self.recording=False
        print(f"Recording saved as recordings/recording_{i}.wav")

    def record_screen_audio(self, audio_filename='audio_output.wav', samplerate=44100, device_name='record+listening'):
        print("Recording screen audio...")
        device_info = sd.query_devices()
        
        # Try to find record+listening first
        try:
            device_index = next(i for i, d in enumerate(device_info) if device_name in d['name'])
            selected_device = device_info[device_index]
            print(f"Found device: {selected_device['name']}")
            print(f"Device info: {selected_device}")
            
            # Check if it has input channels
            if 'max_inputs' in selected_device:
                channels = selected_device['max_inputs']
            else:
                channels = selected_device.get('channels', 0)
            
            print(f"Device input channels: {channels}")
            
            # If record+listening has no input channels, fall back to BlackHole 16ch
            if channels == 0:
                print("record+listening is an output device. Using BlackHole 16ch for recording...")
                device_index = next(i for i, d in enumerate(device_info) if 'BlackHole 16ch' in d['name'])
                selected_device = device_info[device_index]
                print(f"Switched to device: {selected_device['name']}")
                print(f"BlackHole device info: {selected_device}")
                # Limit to 2 channels for compatibility with audio players
                channels = 2
                print(f"Using {channels} channels (limited from 16 for compatibility)")
                
        except StopIteration:
            print("record+listening not found. Using BlackHole 16ch...")
            device_index = next(i for i, d in enumerate(device_info) if 'BlackHole 16ch' in d['name'])
            selected_device = device_info[device_index]
            print(f"Using device: {selected_device['name']}")
            print(f"BlackHole device info: {selected_device}")
            # Limit to 2 channels for compatibility
            channels = 2
            print(f"Using {channels} channels (limited for compatibility)")

        print(f"Final device selection: {selected_device['name']} with {channels} channels")
        
        # Create recordings directory if it doesn't exist
        os.makedirs('recordings', exist_ok=True)
        
        # Find next available filename
        i = 1
        while os.path.exists(f"recordings/screen_recording_{i}.wav"):
            i += 1
        audio_filename = f"recordings/screen_recording_{i}.wav"
        
        start_time = time.time()
        frames = []
        
        try:
            with sd.InputStream(samplerate=samplerate, channels=channels, dtype='int16', device=device_index) as stream:
                print(f"Recording started with device: {selected_device['name']}")
                while self.recording:
                    audio_data, overflowed = stream.read(1024)
                    frames.append(audio_data)
                    
                    # Update timer
                    passed = time.time() - start_time
                    seconds = int(passed % 60)
                    minutes = int(passed // 60)
                    hours = int(passed // 3600)
                    self.label.config(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        except Exception as e:
            print(f"Error during recording: {e}")
            return
        
        if frames:
            # Combine all frames
            audio_data = np.concatenate(frames, axis=0)
            # Ensure the audio data is in the correct format for WAV
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16)
            sf.write(audio_filename, audio_data, samplerate, subtype='PCM_16')
            print(f"Screen recording saved as {audio_filename}")

if __name__ == "__main__":  
    VoiceRecorder()
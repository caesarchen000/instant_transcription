import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper
import speech_recognition as sr
import os
import tempfile
import threading
import time
from googletrans import Translator
from functools import lru_cache

class RealTimeAudioTranslator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Real-Time Audio Translator")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Initialize variables
        self.is_recording = False
        self.sample_rate = 44100
        self.whisper_model = None
        self.translator = Translator()
        
        # Default settings (hardcoded as requested)
        self.use_whisper = True
        self.use_noise_reduction = True
        self.audio_type = "speech"  # speech or music
        
        # Performance optimizations
        self.translation_cache = {}
        self.is_processing = False
        
        # Variables
        self.source_language = tk.StringVar()
        self.target_language = tk.StringVar()
        
        # Load Whisper model
        print("Loading Whisper model...")
        try:
            self.whisper_model = whisper.load_model("tiny")
            print("Whisper model loaded successfully")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            self.whisper_model = None
        
        self.setup_ui()
        self.load_languages()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)
        main_frame.rowconfigure(7, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Real-Time Audio Translator", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Language selection
        lang_frame = ttk.Frame(main_frame)
        lang_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(lang_frame, text="Input Language:").grid(row=0, column=0, sticky=tk.W)
        self.source_lang_combo = ttk.Combobox(lang_frame, textvariable=self.source_language, width=20)
        self.source_lang_combo.grid(row=0, column=1, padx=10)
        
        ttk.Label(lang_frame, text="Output Language:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        self.target_lang_combo = ttk.Combobox(lang_frame, textvariable=self.target_language, width=20)
        self.target_lang_combo.grid(row=0, column=3, padx=10)
        
        # Recording button
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        self.record_button = ttk.Button(button_frame, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT, padx=5)
        
        # Timer display
        self.timer_label = ttk.Label(button_frame, text="00:00:00", font=("Arial", 12, "bold"))
        self.timer_label.pack(side=tk.LEFT, padx=20)
        
        # Progress bar
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, expand=True)
        
        # Transcription area
        ttk.Label(main_frame, text="Transcription:").grid(row=4, column=0, sticky=(tk.W, tk.N), pady=(10,5))
        self.transcription_text = tk.Text(main_frame, height=8, width=80)
        self.transcription_text.grid(row=4, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Spacing between transcription and translation
        ttk.Frame(main_frame, height=20).grid(row=5, column=0, columnspan=3)
        
        # Translation area
        ttk.Label(main_frame, text="Translation:").grid(row=6, column=0, sticky=(tk.W, tk.N), pady=(10,5))
        self.translation_text = tk.Text(main_frame, height=8, width=80)
        self.translation_text.grid(row=6, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready to record")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
    
    def load_languages(self):
        """Load available languages for translation"""
        self.languages = {
            "English": "en",
            "Spanish": "es",
            "French": "fr", 
            "German": "de",
            "Italian": "it",
            "Portuguese": "pt",
            "Russian": "ru",
            "Japanese": "ja",
            "Korean": "ko",
            "Chinese (Simplified)": "zh-cn",
            "Chinese (Traditional)": "zh-tw",
            "Arabic": "ar",
            "Hindi": "hi",
            "Thai": "th",
            "Vietnamese": "vi"
        }
        
        # Create a mapping for display names to codes
        self.language_mapping = {v: k for k, v in self.languages.items()}
        
        # Set up comboboxes with display names
        display_names = list(self.languages.keys())
        self.source_lang_combo['values'] = display_names
        self.target_lang_combo['values'] = display_names
        
        # Set default values
        self.source_language.set("English")
        self.target_language.set("Chinese (Simplified)")
    
    def get_language_code(self, display_name):
        """Convert display name to language code"""
        return self.languages.get(display_name, "en")
    
    def set_processing_state(self, processing):
        """Enable/disable buttons during processing"""
        self.is_processing = processing
        state = "disabled" if processing else "normal"
        self.record_button.config(state=state)
    
    def update_progress(self, value, text=""):
        """Update progress bar and status"""
        self.progress_var.set(value)
        if text:
            self.status_var.set(text)
    
    def toggle_recording(self):
        """Toggle recording on/off"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording audio"""
        self.is_recording = True
        self.record_button.config(text="Stop Recording")
        self.status_var.set("Recording...")
        
        # Clear text areas when starting new recording
        self.clear_text_areas()
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self.record_audio, daemon=True)
        self.recording_thread.start()
        print("Recording started")
    
    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False
        self.record_button.config(text="Start Recording")
        self.timer_label.config(text="00:00:00")
        self.status_var.set("Recording stopped")
        print("Recording stopped")
    
    def clear_text_areas(self):
        """Clear the transcription and translation text areas"""
        self.transcription_text.delete("1.0", tk.END)
        self.translation_text.delete("1.0", tk.END)
    
    def record_audio(self):
        """Record audio from system audio"""
        try:
            print("Starting audio recording...")
            device_info = sd.query_devices()
            
            # Try to find record+listening first
            try:
                device_index = next(i for i, d in enumerate(device_info) if 'record+listening' in d['name'])
                selected_device = device_info[device_index]
                print(f"Found device: {selected_device['name']}")
                print(f"Device info: {selected_device}")
                
                # Check if it has input channels
                if 'max_inputs' in selected_device:
                    input_channels = selected_device['max_inputs']
                else:
                    input_channels = selected_device.get('max_input_channels', 0)
                
                print(f"Device input channels: {input_channels}")
                
                # If record+listening has no input channels, try BlackHole 16ch
                if input_channels == 0:
                    print("record+listening is an output device. Trying BlackHole 16ch...")
                    try:
                        device_index = next(i for i, d in enumerate(device_info) if 'BlackHole 16ch' in d['name'])
                        selected_device = device_info[device_index]
                        print(f"Found BlackHole 16ch: {selected_device['name']}")
                        print(f"BlackHole device info: {selected_device}")
                        
                        # Check BlackHole input channels
                        if 'max_inputs' in selected_device:
                            input_channels = selected_device['max_inputs']
                        else:
                            input_channels = selected_device.get('max_input_channels', 0)
                        
                        print(f"BlackHole input channels: {input_channels}")
                        
                        if input_channels > 0:
                            channels = min(2, input_channels)
                            print(f"Using BlackHole 16ch with {channels} channels")
                        else:
                            print("BlackHole 16ch also has no input channels. Using default device...")
                            # Use default input device
                            device_index = sd.default.device[0]
                            selected_device = device_info[device_index]
                            print(f"Using default device: {selected_device['name']}")
                            print(f"Default device info: {selected_device}")
                            channels = 2
                            print(f"Using default device with {channels} channels")
                    except StopIteration:
                        print("BlackHole 16ch not found. Using default device...")
                        device_index = sd.default.device[0]
                        selected_device = device_info[device_index]
                        print(f"Using default device: {selected_device['name']}")
                        print(f"Default device info: {selected_device}")
                        channels = 2
                        print(f"Using default device with {channels} channels")
                else:
                    channels = min(2, input_channels)
                    print(f"Using record+listening with {channels} input channels for recording")
                    
            except StopIteration:
                print("record+listening not found. Trying BlackHole 16ch...")
                try:
                    device_index = next(i for i, d in enumerate(device_info) if 'BlackHole 16ch' in d['name'])
                    selected_device = device_info[device_index]
                    print(f"Found BlackHole 16ch: {selected_device['name']}")
                    print(f"BlackHole device info: {selected_device}")
                    
                    # Check BlackHole input channels
                    if 'max_inputs' in selected_device:
                        input_channels = selected_device['max_inputs']
                    else:
                        input_channels = selected_device.get('max_input_channels', 0)
                    
                    print(f"BlackHole input channels: {input_channels}")
                    
                    if input_channels > 0:
                        channels = min(2, input_channels)
                        print(f"Using BlackHole 16ch with {channels} channels")
                    else:
                        print("BlackHole 16ch has no input channels. Using default device...")
                        device_index = sd.default.device[0]
                        selected_device = device_info[device_index]
                        print(f"Using default device: {selected_device['name']}")
                        print(f"Default device info: {selected_device}")
                        channels = 2
                        print(f"Using default device with {channels} channels")
                except StopIteration:
                    print("BlackHole 16ch not found. Using default device...")
                    device_index = sd.default.device[0]
                    selected_device = device_info[device_index]
                    print(f"Using default device: {selected_device['name']}")
                    print(f"Default device info: {selected_device}")
                    channels = 2
                    print(f"Using default device with {channels} channels")

            print(f"Final device selection: {selected_device['name']} with {channels} channels")
            
            # Record audio using selected device (like screen_recorder.py)
            frames = []
            start_time = time.time()
            
            with sd.InputStream(samplerate=self.sample_rate, channels=channels, dtype='int16', device=device_index) as stream:
                print(f"Recording started with device: {selected_device['name']}")
                while self.is_recording:
                    audio_data, overflowed = stream.read(1024)
                    frames.append(audio_data)
                    
                    # Debug: Check audio levels every 50 frames
                    if len(frames) % 50 == 0:
                        # Calculate RMS (Root Mean Square) to check audio levels
                        rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
                        print(f"Audio level (RMS): {rms:.2f} (frames: {len(frames)})")
                    
                    # Update timer
                    passed = time.time() - start_time
                    seconds = int(passed % 60)
                    minutes = int(passed // 60)
                    hours = int(passed // 3600)
                    self.root.after(0, lambda: self.timer_label.config(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}"))
                    
                print("Audio recording stopped")
                
                # Process the recorded audio when stopping
                if frames:
                    print(f"Processing {len(frames)} audio frames...")
                    
                    # Debug: Check overall audio levels
                    all_audio = np.concatenate(frames, axis=0)
                    max_level = np.max(np.abs(all_audio))
                    avg_level = np.mean(np.abs(all_audio))
                    print(f"Audio statistics - Max: {max_level}, Avg: {avg_level:.2f}")
                    
                    self.root.after(0, lambda: self.status_var.set("Processing audio..."))
                    self.root.after(0, lambda: self.update_progress(50, "Transcribing audio..."))
                    
                    # Process the audio frames
                    self.process_audio_frames(frames)
                    
                    self.root.after(0, lambda: self.update_progress(100, "Processing complete"))
                    print("Audio processing completed")
                else:
                    print("No audio frames recorded")
                    self.root.after(0, lambda: self.status_var.set("No audio recorded"))
                    
        except Exception as e:
            print(f"Recording error: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.status_var.set(f"Recording error: {e}"))
    
    def process_audio_frames(self, frames):
        """Process audio frames for transcription and translation"""
        if not frames:
            print("No audio frames to process")
            return
        
        try:
            print(f"Processing {len(frames)} audio frames...")
            
            # Combine audio frames
            audio_data = np.concatenate(frames, axis=0)
            print(f"Combined audio data shape: {audio_data.shape}")
            
            # Save to temporary file for processing
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_filename = temp_file.name
                # Ensure the audio data is in the correct format for WAV
                if audio_data.dtype != np.int16:
                    audio_data = (audio_data * 32767).astype(np.int16)
                sf.write(temp_filename, audio_data, self.sample_rate, subtype='PCM_16')
                print(f"Saved audio to temp file: {temp_filename}")
            
            # Also save to a permanent file for checking
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            permanent_filename = f"recorded_audio_{timestamp}.wav"
            sf.write(permanent_filename, audio_data, self.sample_rate, subtype='PCM_16')
            print(f"Saved audio to permanent file: {permanent_filename}")
            
            # Transcribe audio
            source_lang = self.source_language.get()
            print(f"Transcribing with source language: {source_lang}")
            transcribed_text = self.transcribe_audio(temp_filename, source_lang)
            
            # Clean up temp file
            os.unlink(temp_filename)
            print(f"Deleted temp file: {temp_filename}")
            
            if transcribed_text and transcribed_text.strip():
                print(f"Transcribed text: '{transcribed_text}'")
                # Update transcription text
                self.root.after(0, lambda: self.update_transcription(transcribed_text))
                
                # Translate immediately
                print("Starting translation...")
                self.root.after(0, lambda: self.update_progress(75, "Translating..."))
                self.translate_realtime(transcribed_text)
            else:
                print("No transcribed text found")
                self.root.after(0, lambda: self.status_var.set("No speech detected"))
                
        except Exception as e:
            print(f"Audio processing error: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.status_var.set(f"Processing error: {e}"))
    
    def transcribe_audio(self, audio_file_path, source_lang):
        """Transcribe audio using Whisper"""
        try:
            if not self.whisper_model:
                print("Whisper model not available")
                return None
            
            print(f"Starting Whisper transcription for: {audio_file_path}")
            
            # Get language code for Whisper
            whisper_lang = self.get_whisper_language_code(source_lang)
            print(f"Whisper language code: {whisper_lang}")
            
            # Transcribe with Whisper using simpler parameters
            result = self.whisper_model.transcribe(
                audio_file_path,
                language=whisper_lang if whisper_lang != "auto" else None,
                task="transcribe",
                fp16=False,  # Disable FP16 to avoid tensor issues
                verbose=False  # Reduce output
            )
            
            text = result["text"].strip()
            print(f"Whisper result: '{text}'")
            return text if text else None
            
        except Exception as e:
            print(f"Transcription error: {e}")
            # Try with auto language detection as fallback
            try:
                print("Trying with auto language detection...")
                result = self.whisper_model.transcribe(
                    audio_file_path,
                    language=None,  # Auto-detect
                    task="transcribe",
                    fp16=False,
                    verbose=False
                )
                text = result["text"].strip()
                print(f"Whisper auto-detect result: '{text}'")
                return text if text else None
            except Exception as e2:
                print(f"Auto-detect also failed: {e2}")
                return None
    
    def get_whisper_language_code(self, source_lang):
        """Convert language display name to Whisper language code"""
        whisper_lang_map = {
            "auto": "auto",
            "English": "en",
            "Spanish": "es", 
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Portuguese": "pt",
            "Russian": "ru",
            "Japanese": "ja",
            "Korean": "ko",
            "Chinese (Simplified)": "zh",
            "Chinese (Traditional)": "zh",
            "Arabic": "ar",
            "Hindi": "hi",
            "Thai": "th",
            "Vietnamese": "vi"
        }
        return whisper_lang_map.get(source_lang, "auto")
    
    def update_transcription(self, text):
        """Update the transcription text area"""
        current_text = self.transcription_text.get("1.0", tk.END).strip()
        if current_text:
            new_text = current_text + "\n" + text
        else:
            new_text = text
        
        self.transcription_text.delete("1.0", tk.END)
        self.transcription_text.insert("1.0", new_text)
    
    def translate_realtime(self, text):
        """Translate text in real-time"""
        try:
            source_lang = self.source_language.get()
            target_lang = self.target_language.get()
            
            print(f"Translating text: '{text}'")
            print(f"From {source_lang} to {target_lang}")
            
            # Get language codes
            source_code = self.get_language_code(source_lang)
            target_code = self.get_language_code(target_lang)
            
            print(f"Language codes: {source_code} -> {target_code}")
            
            # Translate
            result = self.translator.translate(text, src=source_code, dest=target_code)
            translated_text = result.text
            
            print(f"Translation result: '{translated_text}'")
            
            # Update translation area
            current_translation = self.translation_text.get("1.0", tk.END).strip()
            if current_translation:
                new_translation = current_translation + "\n" + translated_text
            else:
                new_translation = translated_text
            
            self.root.after(0, lambda: self.update_translation(new_translation))
            self.root.after(0, lambda: self.status_var.set("Translation completed"))
            print("Translation updated in UI")
            
        except Exception as e:
            print(f"Real-time translation error: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.status_var.set(f"Translation error: {e}"))
    
    def update_translation(self, text):
        """Update the translation text area"""
        self.translation_text.delete("1.0", tk.END)
        self.translation_text.insert("1.0", text)

if __name__ == "__main__":
    app = RealTimeAudioTranslator()
    app.root.mainloop()

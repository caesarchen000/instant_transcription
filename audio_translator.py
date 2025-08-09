import os
import speech_recognition as sr
from googletrans import Translator
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import json
import time
from functools import lru_cache
import numpy as np
import soundfile as sf
import whisper

class AudioTranslator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Audio Translator")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Initialize recognizer and translator
        self.recognizer = sr.Recognizer()
        self.translator = Translator()
        
        # Initialize Whisper model
        try:
            self.whisper_model = whisper.load_model("base")
            self.whisper_available = True
            print("Whisper model loaded successfully")
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            self.whisper_available = False
        
        # Performance optimizations
        self.translation_cache = {}
        self.is_processing = False
        
        # Transcription settings
        self.noise_reduction = tk.BooleanVar(value=True)
        self.use_multiple_engines = tk.BooleanVar(value=True)
        self.use_whisper = tk.BooleanVar(value=True)
        
        # Variables
        self.selected_file = tk.StringVar()
        self.source_language = tk.StringVar()
        self.target_language = tk.StringVar()
        self.transcription_text = tk.StringVar()
        self.translation_text = tk.StringVar()
        
        self.setup_ui()
        self.load_languages()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)
        main_frame.rowconfigure(7, weight=1)
        
        # File selection
        ttk.Label(main_frame, text="Audio File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.selected_file, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5)
        
        # Language selection
        lang_frame = ttk.Frame(main_frame)
        lang_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(lang_frame, text="Source Language:").grid(row=0, column=0, sticky=tk.W)
        self.source_lang_combo = ttk.Combobox(lang_frame, textvariable=self.source_language, width=20)
        self.source_lang_combo.grid(row=0, column=1, padx=10)
        
        ttk.Label(lang_frame, text="Target Language:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        self.target_lang_combo = ttk.Combobox(lang_frame, textvariable=self.target_language, width=20)
        self.target_lang_combo.grid(row=0, column=3, padx=10)
        
        # Options frame
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=5)
        
        # Options checkboxes
        self.use_whisper = tk.BooleanVar(value=True)
        self.noise_reduction = tk.BooleanVar()
        self.use_multiple_engines = tk.BooleanVar()
        
        ttk.Checkbutton(options_frame, text="Use Whisper (Better accuracy)", variable=self.use_whisper).grid(row=0, column=0, sticky="w", padx=5)
        ttk.Checkbutton(options_frame, text="Noise Reduction", variable=self.noise_reduction).grid(row=0, column=1, sticky="w", padx=5)
        ttk.Checkbutton(options_frame, text="Multiple Engines", variable=self.use_multiple_engines).grid(row=0, column=2, sticky="w", padx=5)
        
        # Audio type selection
        ttk.Label(options_frame, text="Audio Type:").grid(row=1, column=0, sticky="w", padx=5, pady=(10,0))
        self.audio_type = tk.StringVar(value="Speech")
        ttk.Radiobutton(options_frame, text="Speech", variable=self.audio_type, value="Speech").grid(row=1, column=1, sticky="w", padx=5, pady=(10,0))
        ttk.Radiobutton(options_frame, text="Music", variable=self.audio_type, value="Music").grid(row=1, column=2, sticky="w", padx=5, pady=(10,0))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        self.transcribe_btn = ttk.Button(button_frame, text="Transcribe Audio", command=self.transcribe_audio)
        self.transcribe_btn.pack(side=tk.LEFT, padx=5)
        
        self.translate_btn = ttk.Button(button_frame, text="Translate Text", command=self.translate_text)
        self.translate_btn.pack(side=tk.LEFT, padx=5)
        
        self.transcribe_translate_btn = ttk.Button(button_frame, text="Transcribe & Translate", command=self.transcribe_and_translate)
        self.transcribe_translate_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, expand=True)
        
        # Transcription area
        ttk.Label(main_frame, text="Transcription:").grid(row=5, column=0, sticky=(tk.W, tk.N), pady=(10,5))
        self.transcription_text = tk.Text(main_frame, height=8, width=80)
        self.transcription_text.grid(row=5, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Spacing between transcription and translation
        ttk.Frame(main_frame, height=20).grid(row=6, column=0, columnspan=3)
        
        # Translation area
        ttk.Label(main_frame, text="Translation:").grid(row=7, column=0, sticky=(tk.W, tk.N), pady=(10,5))
        self.translation_text = tk.Text(main_frame, height=8, width=80)
        self.translation_text.grid(row=7, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
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
        self.target_language.set("English")
    
    def get_language_code(self, display_name):
        """Convert display name to language code"""
        return self.languages.get(display_name, "auto")
    
    def set_processing_state(self, processing):
        """Enable/disable buttons during processing"""
        self.is_processing = processing
        state = "disabled" if processing else "normal"
        self.transcribe_btn.config(state=state)
        self.translate_btn.config(state=state)
        self.transcribe_translate_btn.config(state=state)
    
    def update_progress(self, value, text=""):
        """Update progress bar and status"""
        self.progress_var.set(value)
        if text:
            self.status_var.set(text)
    
    @lru_cache(maxsize=1000)
    def cached_translate(self, text, source_lang, target_lang):
        """Cached translation for better performance"""
        return self.translator.translate(text, src=source_lang, dest=target_lang)
    
    def fast_translate(self, text, source_lang, target_lang):
        """Fast translation with better performance"""
        try:
            # For short texts, use direct translation
            if len(text) < 500:
                return self.cached_translate(text, source_lang, target_lang)
            
            # For longer texts, split into chunks but keep them larger
            chunks = []
            sentences = text.split('. ')
            
            # Group sentences into larger chunks for better performance
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk + sentence) < 1000:  # Larger chunks
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Translate chunks
            translated_chunks = []
            total_chunks = len(chunks)
            
            for i, chunk in enumerate(chunks):
                try:
                    translated = self.cached_translate(chunk, source_lang, target_lang)
                    translated_chunks.append(translated.text)
                    
                    # Update progress
                    progress = (i + 1) / total_chunks * 100
                    self.root.after(0, lambda p=progress: self.update_progress(p, f"Translating... {int(p)}%"))
                    
                except Exception as e:
                    print(f"Error translating chunk: {e}")
                    translated_chunks.append(chunk)  # Keep original if translation fails
            
            # Combine results
            result_text = ' '.join(translated_chunks)
            return type('TranslationResult', (), {'text': result_text})()
            
        except Exception as e:
            # Fallback to direct translation
            return self.cached_translate(text, source_lang, target_lang)
    
    def batch_translate(self, text, source_lang, target_lang):
        """Translate text in batches for better performance"""
        # Use the improved fast_translate method
        return self.fast_translate(text, source_lang, target_lang)
    
    def browse_file(self):
        """Open file dialog to select audio file"""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.m4a *.flac *.ogg"),
                ("WAV files", "*.wav"),
                ("MP3 files", "*.mp3"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.selected_file.set(file_path)
            self.status_var.set(f"Selected file: {os.path.basename(file_path)}")
    
    def reduce_noise(self, audio_data, sample_rate):
        """Reduce background noise from audio using simple techniques"""
        try:
            # Convert to numpy array if needed
            if hasattr(audio_data, 'get_array_of_samples'):
                audio_array = np.array(audio_data.get_array_of_samples())
            else:
                audio_array = np.array(audio_data)
            
            # Normalize audio
            audio_array = audio_array.astype(np.float32)
            audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Simple noise reduction techniques
            
            # 1. High-pass filter to remove low frequency noise
            from scipy import signal
            nyquist = sample_rate / 2
            cutoff = 80  # Hz - remove frequencies below 80Hz
            b, a = signal.butter(4, cutoff/nyquist, btype='high')
            audio_filtered = signal.filtfilt(b, a, audio_array)
            
            # 2. Simple amplitude thresholding
            threshold = 0.1 * np.max(np.abs(audio_filtered))
            audio_thresholded = np.where(np.abs(audio_filtered) < threshold, 0, audio_filtered)
            
            # 3. Normalize again
            if np.max(np.abs(audio_thresholded)) > 0:
                audio_thresholded = audio_thresholded / np.max(np.abs(audio_thresholded))
            
            return audio_thresholded
            
        except Exception as e:
            print(f"Noise reduction failed: {e}")
            return audio_data
    
    def detect_language(self, text):
        """Detect the language of text using Google Translate"""
        try:
            if not text or len(text.strip()) < 3:
                return "en"  # Default to English for very short text
            
            # Use Google Translate to detect language
            detected = self.translator.detect(text)
            print(f"Language detection result: {detected.lang} (confidence: {detected.confidence})")
            return detected.lang
        except Exception as e:
            print(f"Language detection failed: {e}")
            return "en"  # Default to English
    
    def transcribe_with_whisper(self, audio_file_path, source_lang):
        """Transcribe using OpenAI Whisper"""
        try:
            print(f"Attempting Whisper transcription for: {audio_file_path}")
            
            # Get language code for Whisper
            whisper_lang = self.get_whisper_language_code(source_lang)
            print(f"Whisper language code: {whisper_lang}")
            
            # For auto-detection, use a more robust approach
            if whisper_lang == "auto" or source_lang == "Auto Detect":
                print("Using Whisper with language auto-detection")
                result = self.whisper_model.transcribe(
                    audio_file_path,
                    language=None,  # Let Whisper auto-detect
                    task="transcribe"
                )
                
                text = result["text"].strip()
                detected_lang = result.get("language", "unknown")
                print(f"Whisper auto-detected language: {detected_lang}")
                print(f"Whisper result: {text}")
                
                # If auto-detection failed or detected English for non-English content,
                # try with specific language hints
                if not text or (detected_lang == "en" and len(text) < 10):
                    print("Auto-detection may have failed, trying with language hints...")
                    # Try common languages that might be in the audio
                    languages_to_try = ["zh", "ja", "ko", "es", "fr", "de", "it", "pt", "ru", "ar", "hi", "th", "vi"]
                    
                    for lang in languages_to_try:
                        try:
                            print(f"Trying Whisper with language hint: {lang}")
                            result = self.whisper_model.transcribe(
                                audio_file_path,
                                language=lang,
                                task="transcribe"
                            )
                            text = result["text"].strip()
                            if text and len(text) > 5:
                                print(f"Whisper succeeded with language {lang}: {text}")
                                return text
                        except Exception as e:
                            print(f"Whisper failed with language {lang}: {e}")
                            continue
                    
                    # If all specific languages failed, return the auto-detection result
                    return text if text else "Could not transcribe audio"
                else:
                    return text
            else:
                print(f"Using Whisper with specified language: {whisper_lang}")
                result = self.whisper_model.transcribe(
                    audio_file_path,
                    language=whisper_lang,
                    task="transcribe"
                )
                
                text = result["text"].strip()
                detected_lang = result.get("language", "unknown")
                print(f"Whisper result: {text}")
                print(f"Whisper detected language: {detected_lang}")
                
                return text
            
        except Exception as e:
            print(f"Whisper transcription failed: {e}")
            return None
    
    def transcribe_with_google_fallback(self, audio_file_path, source_lang):
        """Transcribe using Google Speech Recognition with multiple attempts"""
        try:
            print(f"Attempting Google transcription for: {audio_file_path}")
            
            with sr.AudioFile(audio_file_path) as source:
                # Adjust recognition parameters for better results
                self.recognizer.energy_threshold = 4000  # Higher threshold
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.pause_threshold = 0.8
                
                audio = self.recognizer.record(source)
            
            # Try multiple language codes with better ordering
            language_attempts = [
                source_lang,  # Try the specified language first
                "zh-CN",      # Chinese (Simplified)
                "zh-TW",      # Chinese (Traditional)
                "ja-JP",      # Japanese
                "ko-KR",      # Korean
                "es-ES",      # Spanish
                "fr-FR",      # French
                "de-DE",      # German
                "it-IT",      # Italian
                "pt-BR",      # Portuguese
                "ru-RU",      # Russian
                "ar-SA",      # Arabic
                "hi-IN",      # Hindi
                "th-TH",      # Thai
                "vi-VN",      # Vietnamese
                "en-US",      # English (try last for non-English content)
                "auto"        # Auto-detect (try last)
            ]
            
            # Remove duplicates while preserving order
            seen = set()
            unique_attempts = []
            for lang in language_attempts:
                if lang not in seen:
                    seen.add(lang)
                    unique_attempts.append(lang)
            
            for lang in unique_attempts:
                try:
                    print(f"Trying Google recognition with language: {lang}")
                    text = self.recognizer.recognize_google(audio, language=lang)
                    print(f"Google result with {lang}: {text}")
                    if text and text.strip():
                        return text.strip()
                except sr.UnknownValueError:
                    print(f"Google couldn't understand audio with language: {lang}")
                    continue
                except sr.RequestError as e:
                    print(f"Google request error with {lang}: {e}")
                    continue
                except Exception as e:
                    print(f"Google error with {lang}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            print(f"Google transcription failed: {e}")
            return None
    
    def check_audio_file(self, audio_file_path):
        """Check if audio file is valid and readable"""
        try:
            # Check if file exists
            if not os.path.exists(audio_file_path):
                return False, "File does not exist"
            
            # Check file size
            file_size = os.path.getsize(audio_file_path)
            if file_size == 0:
                return False, "File is empty"
            
            # Try to read audio file
            with sr.AudioFile(audio_file_path) as source:
                # Try to get audio duration
                duration = source.DURATION
                if duration < 0.1:  # Less than 0.1 seconds
                    return False, "Audio file is too short"
                
                print(f"Audio file info: {duration}s, {source.SAMPLE_RATE}Hz, {source.SAMPLE_WIDTH} bytes")
                return True, f"Valid audio file: {duration:.2f}s"
                
        except Exception as e:
            return False, f"Audio file error: {str(e)}"
    
    def get_whisper_language_code(self, source_lang):
        """Convert language display name to Whisper language code"""
        # Whisper uses different language codes than Google Translate
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
    
    def extract_vocals_from_music(self, audio_file_path):
        """Try to extract vocals from music for better transcription"""
        try:
            print("Attempting vocal extraction...")
            
            # Load audio with soundfile instead of librosa
            y, sr = sf.read(audio_file_path)
            
            # Convert to mono if stereo
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            
            # Apply vocal enhancement techniques using scipy
            
            # 1. Apply high-pass filter to remove low frequencies (bass, drums)
            from scipy import signal
            nyquist = sr / 2
            cutoff = 200  # Hz - remove frequencies below 200Hz
            b, a = signal.butter(4, cutoff/nyquist, btype='high')
            y_filtered = signal.filtfilt(b, a, y)
            
            # 2. Apply spectral gating to reduce background music
            # Use STFT for spectral processing
            f, t, Zxx = signal.stft(y_filtered, sr, nperseg=1024, noverlap=512)
            
            # Get magnitude spectrum
            magnitude = np.abs(Zxx)
            
            # Estimate noise from first few frames
            noise_estimate = np.mean(magnitude[:, :5], axis=1)
            
            # Apply spectral gating
            gate_threshold = 2.0 * noise_estimate
            gate_threshold = gate_threshold.reshape(-1, 1)
            
            # Apply gate
            magnitude_gated = np.maximum(magnitude - gate_threshold, 0)
            
            # Reconstruct audio
            Zxx_gated = magnitude_gated * np.exp(1j * np.angle(Zxx))
            y_vocals, _ = signal.istft(Zxx_gated, sr)
            
            # Normalize audio
            y_vocals = y_vocals / np.max(np.abs(y_vocals))
            
            # Save vocal-enhanced audio
            temp_vocal_file = "temp_vocals.wav"
            sf.write(temp_vocal_file, y_vocals, sr)
            
            print("Vocal extraction completed")
            return temp_vocal_file
            
        except Exception as e:
            print(f"Vocal extraction failed: {e}")
            return None
    
    def transcribe_music_with_whisper(self, audio_file_path):
        """Transcribe music using Whisper with music-specific settings"""
        try:
            print(f"Attempting Whisper music transcription for: {audio_file_path}")
            
            # Whisper with music-specific settings
            result = self.whisper_model.transcribe(
                audio_file_path,
                language=None,  # Auto-detect language for music
                task="transcribe",
                # Music-specific parameters
                condition_on_previous_text=False,  # Don't condition on previous text
                temperature=0.0,  # More deterministic
                compression_ratio_threshold=2.4,  # Lower threshold for music
                logprob_threshold=-1.0,  # Lower threshold for music
                no_speech_threshold=0.6  # Higher threshold for music
            )
            
            text = result["text"].strip()
            detected_lang = result.get("language", "unknown")
            print(f"Whisper music result: {text}")
            print(f"Whisper music detected language: {detected_lang}")
            
            if text:
                return text
            else:
                return "No lyrics or speech detected in music"
                
        except Exception as e:
            print(f"Whisper music transcription failed: {e}")
            return f"Music transcription failed: {str(e)}"
    
    def transcribe_music_with_google(self, audio_file_path):
        """Transcribe music using Google with music-optimized settings"""
        try:
            print(f"Attempting Google music transcription for: {audio_file_path}")
            
            with sr.AudioFile(audio_file_path) as source:
                # Music-optimized settings
                self.recognizer.energy_threshold = 100  # Much lower for music
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.pause_threshold = 0.3  # Much shorter pauses for music
                self.recognizer.phrase_threshold = 0.1  # Much shorter phrases for music
                self.recognizer.non_speaking_duration = 0.1  # Much shorter non-speaking duration
                
                audio = self.recognizer.record(source)
            
            # Try multiple languages for music, prioritizing non-English languages
            music_languages = [
                "zh-CN",      # Chinese (Simplified)
                "zh-TW",      # Chinese (Traditional)
                "ja-JP",      # Japanese
                "ko-KR",      # Korean
                "es-ES",      # Spanish
                "fr-FR",      # French
                "de-DE",      # German
                "it-IT",      # Italian
                "pt-BR",      # Portuguese
                "ru-RU",      # Russian
                "ar-SA",      # Arabic
                "hi-IN",      # Hindi
                "th-TH",      # Thai
                "vi-VN",      # Vietnamese
                "en-US",      # English (try later for non-English music)
                "auto"        # Auto-detect (try last)
            ]
            
            for lang in music_languages:
                try:
                    print(f"Trying Google music recognition with language: {lang}")
                    text = self.recognizer.recognize_google(audio, language=lang)
                    print(f"Google music result with {lang}: {text}")
                    if text and text.strip():
                        return text.strip()
                except sr.UnknownValueError:
                    print(f"Google couldn't understand music with language: {lang}")
                    continue
                except sr.RequestError as e:
                    print(f"Google request error with {lang}: {e}")
                    continue
                except Exception as e:
                    print(f"Google error with {lang}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            print(f"Google music transcription failed: {e}")
            return None
    
    def transcribe_music(self, audio_file_path):
        """Transcribe music using multiple approaches"""
        print(f"Starting music transcription for: {audio_file_path}")
        
        # Approach 1: Try Whisper first (best for music)
        if self.whisper_available and self.use_whisper.get():
            print("Trying Whisper for music transcription...")
            result = self.transcribe_music_with_whisper(audio_file_path)
            if result and result != "No lyrics or speech detected in music":
                print("Whisper music transcription successful")
                return result
            else:
                print("Whisper failed, trying Google...")
        
        # Approach 2: Try Google with music-optimized settings
        print("Trying Google with music-optimized settings...")
        result = self.transcribe_music_with_google(audio_file_path)
        if result:
            print("Google music transcription successful")
            return result
        
        # Approach 3: Try vocal extraction + transcription
        print("Trying vocal extraction...")
        vocal_file = self.extract_vocals_from_music(audio_file_path)
        if vocal_file:
            try:
                # Try Whisper with extracted vocals
                if self.whisper_available:
                    print("Trying Whisper with extracted vocals...")
                    result = self.whisper_model.transcribe(
                        vocal_file,
                        language=None,  # Auto-detect
                        task="transcribe"
                    )
                    text = result["text"].strip()
                    if text:
                        print("Vocal extraction + Whisper successful")
                        # Clean up temp file
                        if os.path.exists(vocal_file):
                            os.remove(vocal_file)
                        return text
                
                # Try Google with extracted vocals
                print("Trying Google with extracted vocals...")
                with sr.AudioFile(vocal_file) as source:
                    # Use very permissive settings for extracted vocals
                    self.recognizer.energy_threshold = 50
                    self.recognizer.pause_threshold = 0.1
                    audio = self.recognizer.record(source)
                
                for lang in ["auto", "en-US", "zh-CN", "ja-JP", "ko-KR"]:
                    try:
                        print(f"Trying Google with extracted vocals, language: {lang}")
                        text = self.recognizer.recognize_google(audio, language=lang)
                        if text and text.strip():
                            print("Vocal extraction + Google successful")
                            # Clean up temp file
                            if os.path.exists(vocal_file):
                                os.remove(vocal_file)
                            return text.strip()
                    except:
                        continue
                
                # Clean up temp file
                if os.path.exists(vocal_file):
                    os.remove(vocal_file)
                    
            except Exception as e:
                print(f"Vocal extraction transcription failed: {e}")
                # Clean up temp file
                if os.path.exists(vocal_file):
                    os.remove(vocal_file)
        
        # If all approaches failed
        print("All music transcription approaches failed")
        return "No lyrics or speech detected in music"
    
    def transcribe_with_multiple_engines(self, audio, source_lang):
        """Transcribe using multiple engines for better accuracy"""
        results = []
        
        # Engine 1: Google Speech Recognition
        try:
            text1 = self.recognizer.recognize_google(audio, language=source_lang)
            results.append(("Google", text1))
        except Exception as e:
            print(f"Google recognition failed: {e}")
        
        # Engine 2: Google Speech Recognition with auto language detection
        if source_lang == "auto":
            try:
                text2 = self.recognizer.recognize_google(audio)
                results.append(("Google Auto", text2))
            except Exception as e:
                print(f"Google auto recognition failed: {e}")
        
        # Engine 3: Try with different language codes for mixed content
        if len(results) == 0:
            # Try common language combinations
            mixed_languages = ["en-US", "zh-CN", "ja-JP", "ko-KR", "es-ES", "fr-FR", "de-DE"]
            for lang in mixed_languages:
                try:
                    text = self.recognizer.recognize_google(audio, language=lang)
                    results.append((f"Google {lang}", text))
                    break
                except:
                    continue
        
        return results
    
    def merge_transcription_results(self, results):
        """Merge multiple transcription results for better accuracy"""
        if not results:
            return ""
        
        if len(results) == 1:
            return results[0][1]
        
        # For multiple results, choose the longest one (usually most complete)
        # or combine them intelligently
        texts = [result[1] for result in results]
        
        # Choose the longest transcription as it's usually most complete
        longest_text = max(texts, key=len)
        
        # If there are significant differences, show all options
        if len(set(texts)) > 1:
            print("Multiple transcription results detected:")
            for engine, text in results:
                print(f"{engine}: {text}")
        
        return longest_text
    
    def transcribe_audio(self):
        """Transcribe the selected audio file following the process: audio → transcribe → translate"""
        file_path = self.selected_file.get()
        if not file_path:
            messagebox.showerror("Error", "Please select an audio file first.")
            return
        
        # Check if file is valid
        is_valid, message = self.check_audio_file(file_path)
        if not is_valid:
            messagebox.showerror("Error", f"Invalid audio file: {message}")
            return
        
        self.set_processing_state(True)
        self.status_var.set("Transcribing audio...")
        
        def transcribe_thread():
            try:
                source_lang = self.source_language.get()
                audio_type = self.audio_type.get()
                
                print("=" * 50)
                print("STARTING TRANSCRIPTION PROCESS")
                print("=" * 50)
                print(f"Audio file: {file_path}")
                print(f"Audio type: {audio_type}")
                print(f"Selected source language: {source_lang}")
                print(f"Use Whisper: {self.use_whisper.get()}")
                print(f"Noise reduction: {self.noise_reduction.get()}")
                print("=" * 50)
                
                # TRANSCRIBE
                print("\nTRANSCRIBING AUDIO")
                print("-" * 30)
                self.root.after(0, lambda: self.status_var.set("Transcribing audio..."))
                
                result = None
                
                # Use Whisper approach for transcription
                if self.whisper_available and self.use_whisper.get():
                    print("Using Whisper for transcription...")
                    result = self.transcribe_with_improved_whisper(file_path, source_lang)
                    if result:
                        print(f"✓ Whisper transcription successful")
                    else:
                        print("✗ Whisper transcription failed")
                
                # Fallback to other methods if Whisper failed
                if not result:
                    if audio_type == "Music":
                        print("Trying music-specific transcription...")
                        result = self.transcribe_music(file_path)
                    else:
                        print("Trying Google Speech Recognition...")
                        result = self.transcribe_with_multiple_engines(file_path, source_lang)
                
                if result:
                    print(f"✓ Transcription completed successfully")
                    print(f"Transcribed text: {result[:100]}...")
                    self.root.after(0, lambda: self.update_transcription(result))
                    self.root.after(0, lambda: self.status_var.set("Transcription completed - ready for translation"))
                else:
                    print("✗ All transcription methods failed")
                    self.root.after(0, lambda: messagebox.showerror("Error", "Could not transcribe audio. Please try again."))
                    self.root.after(0, lambda: self.status_var.set("Transcription failed"))
                    return
                
                print("\nREADY FOR TRANSLATION")
                print("-" * 30)
                print("✓ Audio has been transcribed successfully")
                print("✓ You can now click 'Translate' to translate the text")
                print("=" * 50)
                    
            except Exception as e:
                print(f"✗ Transcription error: {e}")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Transcription failed: {str(e)}"))
                self.root.after(0, lambda: self.status_var.set("Transcription failed"))
            finally:
                self.root.after(0, lambda: self.set_processing_state(False))
        
        threading.Thread(target=transcribe_thread, daemon=True).start()
    
    def translate_text(self):
        """Translate the transcribed text (Step 4 of the process)"""
        text = self.transcription_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Error", "Please transcribe audio first.")
            return
        
        target_lang = self.target_language.get()
        source_lang = self.source_language.get()
        
        if not target_lang or target_lang == "Select Target Language":
            messagebox.showerror("Error", "Please select a target language.")
            return
        
        self.set_processing_state(True)
        self.status_var.set("Step 4: Translating text...")
        
        def translate_thread():
            try:
                print("\n" + "=" * 50)
                print("STEP 4: TRANSLATING TEXT")
                print("=" * 50)
                print(f"Source text: {text[:100]}...")
                print(f"Source language: {source_lang}")
                print(f"Target language: {target_lang}")
                print("-" * 30)
                
                # Get language codes
                source_code = self.get_language_code(source_lang)
                target_code = self.get_language_code(target_lang)
                
                print(f"Source code: {source_code}")
                print(f"Target code: {target_code}")
                
                # Translate using fast translation
                print("Translating...")
                result = self.fast_translate(text, source_code, target_code)
                
                if result and hasattr(result, 'text'):
                    translated_text = result.text
                    print(f"✓ Translation completed successfully")
                    print(f"Translated text: {translated_text[:100]}...")
                    self.root.after(0, lambda: self.update_translation(translated_text))
                    self.root.after(0, lambda: self.status_var.set("Translation completed"))
                else:
                    print("✗ Translation failed")
                    self.root.after(0, lambda: messagebox.showerror("Error", "Translation failed. Please try again."))
                    self.root.after(0, lambda: self.status_var.set("Translation failed"))
                    return
                
                print("\nCOMPLETE PROCESS SUMMARY:")
                print("-" * 30)
                print("✓ Step 1: Audio file analyzed")
                print("✓ Step 2: Language detected")
                print("✓ Step 3: Audio transcribed")
                print("✓ Step 4: Text translated")
                print("=" * 50)
                
            except Exception as e:
                print(f"✗ Translation error: {e}")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Translation failed: {str(e)}"))
                self.root.after(0, lambda: self.status_var.set("Translation failed"))
            finally:
                self.root.after(0, lambda: self.set_processing_state(False))
        
        threading.Thread(target=translate_thread, daemon=True).start()
    
    def transcribe_and_translate(self):
        """Transcribe audio and then translate the result"""
        if not self.selected_file.get():
            messagebox.showerror("Error", "Please select an audio file first.")
            return
        
        if self.is_processing:
            return
        
        def process_thread():
            try:
                self.set_processing_state(True)
                self.update_progress(0, "Loading audio file...")
                
                # Load the audio file
                with sr.AudioFile(self.selected_file.get()) as source:
                    self.update_progress(15, "Processing audio...")
                    audio = self.recognizer.record(source)
                
                # Get language code from display name
                source_lang_code = self.get_language_code(self.source_language.get())
                
                self.update_progress(30, "Transcribing audio...")
                
                # Perform speech recognition
                text = self.recognizer.recognize_google(audio, language=source_lang_code)
                
                # Update transcription in main thread
                self.root.after(0, lambda: self.update_transcription(text))
                
                self.update_progress(50, "Starting translation...")
                
                # Get language codes from display names
                source_lang = self.get_language_code(self.source_language.get())
                target_lang = self.get_language_code(self.target_language.get())
                
                # Handle special cases for Chinese
                if target_lang == "zh":
                    target_lang = "zh-cn"  # Default to simplified Chinese
                
                self.update_progress(60, "Translating text...")
                
                # Use batch translation for better performance
                translation = self.batch_translate(text, source_lang, target_lang)
                
                self.update_progress(100, "Transcription and translation completed")
                
                # Update translation in main thread
                self.root.after(0, lambda: self.update_translation(translation.text))
                self.root.after(0, lambda: self.set_processing_state(False))
                
            except sr.UnknownValueError:
                self.root.after(0, lambda: messagebox.showerror("Error", "Could not understand the audio"))
                self.root.after(0, lambda: self.set_processing_state(False))
            except sr.RequestError as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Could not request results: {e}"))
                self.root.after(0, lambda: self.set_processing_state(False))
            except Exception as e:
                error_msg = f"Error during processing: {str(e)}"
                if "invalid destination language" in str(e).lower():
                    error_msg = "Invalid language code. Please select a valid language from the dropdown."
                self.root.after(0, lambda: messagebox.showerror("Processing Error", error_msg))
                self.root.after(0, lambda: self.set_processing_state(False))
        
        threading.Thread(target=process_thread, daemon=True).start()
    
    def update_transcription(self, text):
        """Update the transcription textbox"""
        self.transcription_text.delete("1.0", tk.END)
        self.transcription_text.insert("1.0", text)
    
    def update_translation(self, text):
        """Update the translation textbox"""
        self.translation_text.delete("1.0", tk.END)
        self.translation_text.insert("1.0", text)
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

    def transcribe_with_improved_whisper(self, audio_file_path, source_lang):
        """Transcribe using Whisper with improved language detection"""
        try:
            print(f"Attempting improved Whisper transcription for: {audio_file_path}")
            
            # Transcribe with specified language
            whisper_lang = self.get_whisper_language_code(source_lang)
            if whisper_lang == "auto":
                result = self.whisper_model.transcribe(
                    audio_file_path,
                    language=None,  # Auto-detect
                    task="transcribe"
                )
            else:
                result = self.whisper_model.transcribe(
                    audio_file_path,
                    language=whisper_lang,
                    task="transcribe"
                )
            
            text = result["text"].strip()
            print(f"Improved Whisper result: {text}")
            
            return text if text else None
            
        except Exception as e:
            print(f"Improved Whisper transcription failed: {e}")
            return None

if __name__ == "__main__":
    app = AudioTranslator()
    app.run()

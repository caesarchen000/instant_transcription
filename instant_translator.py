import tkinter as tk
from tkinter import ttk, messagebox
import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper
import speech_recognition as sr
import os
import tempfile
import threading
import time
import queue
from googletrans import Translator
from functools import lru_cache

class InstantAudioTranslator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Enhanced Instant Audio Translator with Floating Subtitles")
        self.root.geometry("800x700")
        self.root.resizable(True, True)
        
        # Initialize variables
        self.is_recording = False
        self.sample_rate = 16000  # Lower sample rate for faster processing
        self.whisper_model = None
        self.translator = Translator()
        
        # Audio processing queues and buffers
        self.audio_queue = queue.Queue()
        self.processing_queue = queue.Queue()
        self.audio_buffer = []
        self.buffer_size = 64000  # 4 seconds at 16kHz for better context and translation quality
        
        # Context management for better translation quality
        self.context_keywords = ""
        self.previous_transcription = ""
        self.context_window = []  # Keep last few transcriptions for context
        self.max_context_length = 3  # Keep last 3 transcriptions for context
        
        # Processing threads
        self.recording_thread = None
        self.processing_thread = None
        
        # Performance settings
        self.use_whisper = True
        self.use_noise_reduction = True
        self.audio_type = "speech"
        
        # Translation cache
        self.translation_cache = {}
        
        # Variables
        self.source_language = tk.StringVar()
        self.target_language = tk.StringVar()
        
        # Floating subtitle window
        self.floating_window = None
        self.show_floating_subtitles = tk.BooleanVar(value=False)
        
        # Load Whisper model (tiny for speed)
        print("Loading Whisper model...")
        try:
            self.whisper_model = whisper.load_model("tiny")
            print("Whisper model loaded successfully")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            self.whisper_model = None
        
        self.setup_ui()
        self.load_languages()
        
        # Start processing thread
        self.start_processing_thread()
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def on_closing(self):
        """Handle application closing"""
        try:
            # Stop recording if active
            if self.is_recording:
                self.stop_recording()
            
            # Destroy floating subtitle window
            if self.floating_window:
                self.destroy_floating_window()
            
            # Close main window
            self.root.destroy()
            print("Application closed successfully")
        except Exception as e:
            print(f"Error during shutdown: {e}")
            self.root.destroy()
    
    def toggle_floating_subtitles(self):
        """Toggle floating subtitle window on/off"""
        if self.show_floating_subtitles.get():
            self.create_floating_window()
            self.floating_status.config(text="Floating subtitles: Enabled", foreground="green")
        else:
            self.destroy_floating_window()
            self.floating_status.config(text="Floating subtitles: Disabled", foreground="gray")
    
    def create_floating_window(self):
        """Create floating subtitle window"""
        if self.floating_window is None:
            self.floating_window = tk.Toplevel(self.root)
            self.floating_window.title("Floating Subtitles")
            self.floating_window.geometry("600x150")
            self.floating_window.attributes('-topmost', True)
            self.floating_window.overrideredirect(True)
            self.floating_window.configure(bg='black')
            
            # Position at bottom center of screen
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            x = (screen_width - 600) // 2
            y = screen_height - 200
            self.floating_window.geometry(f"600x150+{x}+{y}")
            
            # Subtitle text
            self.floating_text = tk.Text(self.floating_window, bg='black', fg='white', 
                                       font=('Arial', 16), wrap=tk.WORD, height=6)
            self.floating_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Make window draggable
            self.floating_window.bind('<Button-1>', self.start_drag)
            self.floating_window.bind('<B1-Motion>', self.on_drag)
            self.floating_window.bind('<Double-Button-1>', self.toggle_floating_subtitles)
            
            print("Floating subtitle window created")
    
    def destroy_floating_window(self):
        """Destroy floating subtitle window"""
        if self.floating_window:
            self.floating_window.destroy()
            self.floating_window = None
            print("Floating subtitle window destroyed")
    
    def start_drag(self, event):
        """Start dragging the floating window"""
        self.floating_window.x = event.x
        self.floating_window.y = event.y
    
    def on_drag(self, event):
        """Handle dragging of the floating window"""
        deltax = event.x - self.floating_window.x
        deltay = event.y - self.floating_window.y
        x = self.floating_window.winfo_x() + deltax
        y = self.floating_window.winfo_y() + deltay
        self.floating_window.geometry(f"+{x}+{y}")
    
    def update_floating_subtitles(self, original_text, translated_text):
        """Update floating subtitle window with latest text"""
        if self.floating_window and self.show_floating_subtitles.get():
            try:
                # Keep only the last 2 lines for better readability
                lines = []
                if original_text:
                    lines.append(f"Original: {original_text}")
                if translated_text:
                    lines.append(f"Translated: {translated_text}")
                
                # Limit to last 2 lines
                if len(lines) > 2:
                    lines = lines[-2:]
                
                display_text = "\n".join(lines)
                self.floating_text.delete(1.0, tk.END)
                self.floating_text.insert(1.0, display_text)
            except Exception as e:
                print(f"Error updating floating subtitles: {e}")
    
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
        title_label = ttk.Label(main_frame, text="Instant Audio Translator", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Context keywords
        context_frame = ttk.Frame(main_frame)
        context_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(context_frame, text="Context Keywords:").grid(row=0, column=0, sticky=tk.W)
        self.context_keywords_var = tk.StringVar()
        self.context_keywords_entry = ttk.Entry(context_frame, textvariable=self.context_keywords_var, width=50)
        self.context_keywords_entry.grid(row=0, column=1, padx=10, sticky=(tk.W, tk.E))
        self.context_keywords_entry.insert(0, "general conversation")
        
        # Language selection
        lang_frame = ttk.Frame(main_frame)
        lang_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(lang_frame, text="Input Language:").grid(row=0, column=0, sticky=tk.W)
        self.source_lang_combo = ttk.Combobox(lang_frame, textvariable=self.source_language, width=20)
        self.source_lang_combo.grid(row=0, column=1, padx=10)
        
        ttk.Label(lang_frame, text="Output Language:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        self.target_lang_combo = ttk.Combobox(lang_frame, textvariable=self.target_language, width=20)
        self.target_lang_combo.grid(row=0, column=3, padx=10)
        
        # Recording button
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        self.record_button = ttk.Button(button_frame, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT, padx=5)
        
        # Floating subtitle toggle
        self.floating_check = ttk.Checkbutton(button_frame, text="🖥️ Floating Subtitles", 
                                            variable=self.show_floating_subtitles, 
                                            command=self.toggle_floating_subtitles)
        self.floating_check.pack(side=tk.LEFT, padx=20)
        
        # Timer display
        self.timer_label = ttk.Label(button_frame, text="00:00:00", font=("Arial", 12, "bold"))
        self.timer_label.pack(side=tk.LEFT, padx=20)
        
        # Progress bar
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, expand=True)
        
        # Transcription area
        ttk.Label(main_frame, text="Transcription:").grid(row=4, column=0, sticky=tk.W, pady=(10,5))
        self.transcription_text = tk.Text(main_frame, height=8, width=80)
        self.transcription_text.grid(row=4, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Spacing between transcription and translation
        ttk.Frame(main_frame, height=20).grid(row=5, column=0, columnspan=3)
        
        # Translation area
        ttk.Label(main_frame, text="Translation:").grid(row=6, column=0, sticky=tk.W, pady=(10,5))
        self.translation_text = tk.Text(main_frame, height=8, width=80)
        self.translation_text.grid(row=6, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready to record")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Real-time indicator
        self.realtime_indicator = ttk.Label(main_frame, text="", foreground="green", font=("Arial", 9))
        self.realtime_indicator.grid(row=8, column=0, columnspan=3, pady=2)
        
        # Floating subtitle status
        self.floating_status = ttk.Label(main_frame, text="Floating subtitles: Disabled", foreground="gray", font=("Arial", 9))
        self.floating_status.grid(row=9, column=0, columnspan=3, pady=2)
    
    def load_languages(self):
        """Load available languages"""
        languages = {
            "English": "en",
            "Chinese (Simplified)": "zh-cn",
            "Chinese (Traditional)": "zh-tw",
            "Japanese": "ja",
            "Korean": "ko",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Portuguese": "pt",
            "Russian": "ru",
            "Arabic": "ar",
            "Hindi": "hi",
            "Thai": "th",
            "Vietnamese": "vi"
        }
        
        self.source_lang_combo['values'] = list(languages.keys())
        self.target_lang_combo['values'] = list(languages.keys())
        
        # Set default values
        self.source_language.set("English")
        self.target_language.set("Chinese (Traditional)")
    
    def get_language_code(self, display_name):
        """Convert display language name to language code"""
        language_map = {
            "English": "en",
            "Chinese (Simplified)": "zh-cn",
            "Chinese (Traditional)": "zh-tw",
            "Japanese": "ja",
            "Korean": "ko",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Portuguese": "pt",
            "Russian": "ru",
            "Arabic": "ar",
            "Hindi": "hi",
            "Thai": "th",
            "Vietnamese": "vi"
        }
        
        # Get the language code
        language_code = language_map.get(display_name)
        
        if language_code is None:
            # If language not found, show error and use the original display name
            print(f"⚠️ Warning: Unknown language '{display_name}'. Using display name as fallback.")
            # Try to extract a reasonable code from the display name
            if "chinese" in display_name.lower():
                return "zh"  # Generic Chinese
            elif "japanese" in display_name.lower():
                return "ja"
            elif "korean" in display_name.lower():
                return "ko"
            else:
                # For unknown languages, return None so we can handle it properly
                return None
        
        return language_code
    
    def set_processing_state(self, processing):
        """Enable/disable buttons during processing"""
        state = "disabled" if processing else "normal"
        self.record_button.config(state=state)
    
    def update_progress(self, value, text=""):
        """Update progress bar and status"""
        self.progress_var.set(value)
        if text:
            self.status_var.set(text)
    
    def update_realtime_indicator(self, is_processing):
        """Update the real-time processing indicator"""
        if is_processing:
            self.realtime_indicator.config(text="🔄 Processing...", foreground="orange")
        else:
            self.realtime_indicator.config(text="✅ Ready", foreground="green")
    
    def show_processing_stats(self):
        """Show current processing statistics"""
        buffer_time = self.buffer_size / self.sample_rate
        chunk_time = 512 / self.sample_rate
        total_latency = buffer_time + chunk_time
        
        stats_text = f"Buffer: {buffer_time:.1f}s | Chunk: {chunk_time*1000:.1f}ms | Total Latency: {total_latency:.1f}s | 4s chunks for better translation"
        self.realtime_indicator.config(text=stats_text, foreground="blue")
    
    def start_processing_thread(self):
        """Start the background processing thread"""
        self.processing_thread = threading.Thread(target=self.process_audio_continuously, daemon=True)
        self.processing_thread.start()
        print("Processing thread started")
    
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
        
        # Show processing statistics
        self.show_processing_stats()
        
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
    
    def record_audio(self):
        """Record audio continuously and add to processing queue"""
        try:
            print("Starting continuous audio recording...")
            device_info = sd.query_devices()
            
            # Find audio input device
            try:
                device_index = next(i for i, d in enumerate(device_info) if 'BlackHole 16ch' in d['name'])
                selected_device = device_info[device_index]
                channels = min(2, selected_device.get('max_inputs', 2))
                print(f"Using BlackHole 16ch with {channels} channels")
            except StopIteration:
                try:
                    device_index = next(i for i, d in enumerate(device_info) if 'record+listening' in d['name'])
                    selected_device = device_info[device_index]
                    channels = min(2, selected_device.get('max_inputs', 2))
                    print(f"Using record+listening with {channels} channels")
                except StopIteration:
                    device_index = sd.default.device[0]
                    selected_device = device_info[device_index]
                    channels = 2
                    print(f"Using default device: {selected_device['name']}")
            
            # Use smaller chunks for real-time processing
            chunk_size = 512  # Smaller chunks for more frequent updates
            start_time = time.time()
            
            with sd.InputStream(samplerate=self.sample_rate, channels=channels, dtype='int16', device=device_index, blocksize=chunk_size) as stream:
                print(f"Recording started with device: {selected_device['name']}")
                
                while self.is_recording:
                    try:
                        audio_data, overflowed = stream.read(chunk_size)
                        if overflowed:
                            print("Audio overflow detected")
                        
                        # Add to buffer
                        self.audio_buffer.append(audio_data)
                        
                        # Update timer
                        passed = time.time() - start_time
                        seconds = int(passed % 60)
                        minutes = int(passed // 60)
                        hours = int(passed // 3600)
                        self.root.after(0, lambda: self.timer_label.config(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}"))
                        
                        # When buffer is full, send to processing queue
                        if len(self.audio_buffer) * chunk_size >= self.buffer_size:
                            # Combine buffer and send to queue
                            combined_audio = np.concatenate(self.audio_buffer, axis=0)
                            self.audio_queue.put(combined_audio.copy())
                            
                            # Keep overlap for better context continuity (like GitHub repo)
                            overlap_samples = int(self.buffer_size * 0.2)  # 20% overlap (0.8 seconds)
                            overlap_frames = max(1, overlap_samples // chunk_size)
                            self.audio_buffer = self.audio_buffer[-overlap_frames:]
                            
                            print(f"Sent {combined_audio.shape[0]} samples to processing queue (with {overlap_samples} overlap)")
                        
                        # Smaller delay for more responsive updates
                        time.sleep(0.0005)
                        
                    except Exception as e:
                        print(f"Error reading audio: {e}")
                        break
                
                print("Recording stopped")
                
        except Exception as e:
            print(f"Recording error: {e}")
            import traceback
            traceback.print_exc()
    
    def process_audio_continuously(self):
        """Continuously process audio from the queue"""
        print("Audio processing thread started")
        
        while True:
            try:
                # Wait for audio data
                audio_data = self.audio_queue.get(timeout=1)
                
                if audio_data is not None:
                    print(f"Processing {audio_data.shape[0]} samples...")
                    
                    # Process audio immediately
                    self.process_audio_chunk(audio_data)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                import traceback
                traceback.print_exc()
    
    def process_audio_chunk(self, audio_data):
        """Process a single audio chunk for instant transcription"""
        try:
            # Show processing indicator
            self.root.after(0, lambda: self.update_realtime_indicator(True))
            
            # Update progress
            self.root.after(0, lambda: self.update_progress(25, "Processing audio..."))
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_filename = temp_file.name
                sf.write(temp_filename, audio_data, self.sample_rate, subtype='PCM_16')
            
            # Update progress
            self.root.after(0, lambda: self.update_progress(50, "Transcribing audio..."))
            
            # Transcribe with Whisper
            if self.whisper_model:
                try:
                    # Get source language code
                    source_lang_code = self.get_language_code(self.source_language.get())
                    
                    if source_lang_code is None:
                        # If we can't determine the language, show error and skip
                        error_msg = f"⚠️ Cannot determine language for '{self.source_language.get()}'. Please select a valid language."
                        print(error_msg)
                        self.root.after(0, lambda: self.update_transcription(f"[ERROR] {error_msg}"))
                        return
                    
                    result = self.whisper_model.transcribe(temp_filename, fp16=False, language=source_lang_code)
                    transcribed_text = result['text'].strip()
                    
                    if transcribed_text:
                        print(f"Transcribed: {transcribed_text}")
                        
                        # Update context window for better translation
                        self.update_context_window(transcribed_text)
                        
                        # Update progress
                        self.root.after(0, lambda: self.update_progress(75, "Translating..."))
                        
                        # Update transcription in UI with timestamp
                        timestamp = time.strftime("%H:%M:%S")
                        self.root.after(0, lambda: self.update_transcription(f"[{timestamp}] {transcribed_text}"))
                        
                        # Update floating subtitles with original text
                        self.root.after(0, lambda: self.update_floating_subtitles(transcribed_text, ""))
                        
                        # Translate immediately in parallel
                        self.translate_and_update(transcribed_text)
                    
                except Exception as e:
                    print(f"Whisper transcription error: {e}")
            
            # Clean up temp file
            os.unlink(temp_filename)
            
            # Update progress and hide processing indicator
            self.root.after(0, lambda: self.update_progress(100, "Processing complete"))
            self.root.after(0, lambda: self.update_realtime_indicator(False))
            
        except Exception as e:
            print(f"Audio chunk processing error: {e}")
            self.root.after(0, lambda: self.update_progress(0, "Processing error"))
            self.root.after(0, lambda: self.update_realtime_indicator(False))
    
    def translate_and_update(self, text):
        """Translate text and update UI with improved context handling"""
        try:
            source_lang = self.get_language_code(self.source_language.get())
            target_lang = self.get_language_code(self.target_language.get())
            
            # Validate language codes
            if source_lang is None:
                error_msg = f"⚠️ Cannot determine source language for '{self.source_language.get()}'. Please select a valid language."
                print(error_msg)
                self.root.after(0, lambda: self.update_translation(f"[ERROR] {error_msg}"))
                return
                
            if target_lang is None:
                error_msg = f"⚠️ Cannot determine target language for '{self.target_language.get()}'. Please select a valid language."
                print(error_msg)
                self.root.after(0, lambda: self.update_translation(f"[ERROR] {error_msg}"))
                return
            
            print(f"Translating from {source_lang} to {target_lang}")
            
            # Check cache first
            cache_key = f"{text}_{source_lang}_{target_lang}"
            if cache_key in self.translation_cache:
                translated_text = self.translation_cache[cache_key]
                print(f"Cached translation: {translated_text}")
                # Show instant translation indicator
                self.root.after(0, lambda: self.realtime_indicator.config(text="⚡ Instant (cached)", foreground="green"))
            else:
                # Show translation in progress
                self.root.after(0, lambda: self.realtime_indicator.config(text="🌐 Translating...", foreground="purple"))
                
                # Prepare context for better translation quality
                context_prompt = self.prepare_translation_context(text)
                
                # Translate using googletrans with context
                result = self.translator.translate(text, src=source_lang, dest=target_lang)
                translated_text = result.text
                
                # Cache the result
                self.translation_cache[cache_key] = translated_text
                print(f"New translation: {translated_text}")
                
                # Show translation complete
                self.root.after(0, lambda: self.realtime_indicator.config(text="✅ Translation complete", foreground="green"))
            
            # Update translation in UI with timestamp
            timestamp = time.strftime("%H:%M:%S")
            self.root.after(0, lambda: self.update_translation(f"[{timestamp}] {translated_text}"))
            
            # Update floating subtitles
            self.root.after(0, lambda: self.update_floating_subtitles(text, translated_text))
            
        except Exception as e:
            print(f"Translation error: {e}")
            self.root.after(0, lambda: self.realtime_indicator.config(text="❌ Translation error", foreground="red"))
    
    def prepare_translation_context(self, text):
        """Prepare context for better translation quality"""
        context = []
        
        # Add context keywords if available
        if self.context_keywords_var.get().strip():
            context.append(f"Context: {self.context_keywords_var.get()}")
        
        # Add previous transcriptions for continuity
        if self.context_window:
            context.append(f"Previous context: {' '.join(self.context_window[-2:])}")
        
        # Add current text
        context.append(f"Current text: {text}")
        
        return " | ".join(context)
    
    def update_context_window(self, text):
        """Update the context window with new transcription"""
        self.context_window.append(text)
        if len(self.context_window) > self.max_context_length:
            self.context_window.pop(0)
    
    def clear_text_areas(self):
        """Clear the transcription and translation text areas"""
        self.transcription_text.delete("1.0", tk.END)
        self.translation_text.delete("1.0", tk.END)
    
    def update_transcription(self, text):
        """Update transcription text area"""
        self.transcription_text.insert(tk.END, text + "\n")
        self.transcription_text.see(tk.END)
        
        # Limit text area size
        lines = self.transcription_text.get("1.0", tk.END).split('\n')
        if len(lines) > 50:
            self.transcription_text.delete("1.0", "2.0")
    
    def update_translation(self, text):
        """Update translation text area"""
        self.translation_text.insert(tk.END, text + "\n")
        self.translation_text.see(tk.END)
        
        # Limit text area size
        lines = self.translation_text.get("1.0", tk.END).split('\n')
        if len(lines) > 50:
            self.translation_text.delete("1.0", "2.0")
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = InstantAudioTranslator()
    app.run()

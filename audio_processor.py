#!/usr/bin/env python3
"""
Audio Processor for Subtitle Generator
Step 3: Audio recording, Whisper transcription, and translation
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper
import threading
import queue
import time
import json
import requests
from datetime import datetime

import tempfile
import os

class AudioProcessor:
    def __init__(self, server_url='http://localhost:8766'):
        self.server_url = server_url
        self.sample_rate = 16000  # Whisper works best with 16kHz
        self.chunk_size = 1024
        self.buffer_size = 1 * self.sample_rate  # 1 second of audio for faster response
        self.overlap_ratio = 0.2  # 20% overlap between chunks
        
        # Audio processing
        self.audio_buffer = []
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.recording_thread = None
        self.processing_thread = None
        
        # Device info (will be set when recording starts)
        self.input_device = None
        self.input_device_name = None
        
        # Language mapping for deep-translator compatibility
        self.languages = {
            "en": "en",
            "es": "es",
            "fr": "fr", 
            "de": "de",
            "it": "it",
            "pt": "pt",
            "ru": "ru",
            "ja": "ja",
            "ko": "ko",
            "zh": "zh-CN",  # deep-translator expects zh-CN
            "zh-cn": "zh-CN",  # deep-translator expects zh-CN
            "zh-tw": "zh-TW",  # deep-translator expects zh-TW
            "ar": "ar",
            "hi": "hi",
            "th": "th",
            "vi": "vi"
        }
        
        # Whisper model
        print("🧠 Loading Whisper model...")
        try:
            self.whisper_model = whisper.load_model("base")  # Start with base model
            print("✅ Whisper model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading Whisper model: {e}")
            print("💡 Make sure you have Whisper installed: pip install openai-whisper")
            self.whisper_model = None
        
        # Translation - Use googletrans like the working files
        self.translator = None
        self.translation_method = "none"
        
        # Try googletrans first (like text_translator.py and audio_translator.py)
        try:
            from googletrans import Translator
            self.translator = Translator()
            self.translation_method = "googletrans"
            print("✅ Google Translator initialized successfully")
        except ImportError as e:
            print(f"⚠️ googletrans not available: {e}")
        except Exception as e:
            print(f"⚠️ googletrans failed: {e}")
        
        # If googletrans failed, try deep-translator as backup
        if not self.translator:
            try:
                from deep_translator import GoogleTranslator
                # Create an instance for testing with correct language code
                test_translator = GoogleTranslator(source='en', target='zh-CN')
                test_result = test_translator.translate("hello")
                if test_result == "你好":
                    self.translator = "deep_translator"  # Store method identifier
                    self.translation_method = "deep_translator"
                    print("✅ Deep Translator initialized successfully")
                else:
                    print(f"⚠️ Deep translator test failed: expected '你好', got '{test_result}'")
            except Exception as e:
                print(f"⚠️ deep-translator failed: {e}")
        
        # If both failed, use fallback method
        if not self.translator:
            self.translation_method = "fallback"
            print("⚠️ Using fallback translation method")
            print("💡 For better translation, try: pip install googletrans==4.0.0rc1")
        
        # Subtitle context
        self.last_subtitle = ""
        self.context_window = []
        self.max_context_length = 100
        
        print("🎤 Audio processor initialized")
        print(f"🔍 Translation method: {self.translation_method}")
        if self.translator:
            print(f"🔍 Translator type: {type(self.translator)}")
    
    def get_language_code(self, lang_code):
        """Convert language code to googletrans format"""
        return self.languages.get(lang_code, lang_code)
    
    def start_recording(self, input_lang='en', output_lang='zh'):
        """Start audio recording and processing"""
        if self.is_recording:
            print("⚠️ Already recording!")
            return False
        
        if not self.whisper_model:
            print("❌ Whisper model not loaded. Cannot start recording.")
            return False
        
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.is_recording = True
        self.audio_buffer = []
        self.context_window = []
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Update server status
        self._update_server_status(True)
        
        print(f"🎙️ Recording started - Input: {input_lang}, Output: {output_lang}")
        return True
    
    def stop_recording(self):
        """Stop audio recording and processing"""
        if not self.is_recording:
            print("⚠️ Not recording!")
            return False
        
        self.is_recording = False
        
        # Wait for threads to finish
        if self.recording_thread:
            self.recording_thread.join(timeout=2)
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        
        # Update server status
        self._update_server_status(False)
        
        print("⏹️ Recording stopped")
        return True
    
    def _record_audio(self):
        """Record system audio like screen_recorder.py does"""
        try:
            print("🎤 Recording system audio (YouTube, etc.)...")
            device_info = sd.query_devices()
            
            # Try to find record+listening first (like screen_recorder.py)
            try:
                device_index = next(i for i, d in enumerate(device_info) if 'record+listening' in d['name'])
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
            
            # Store device info for server status
            self.input_device = device_index
            self.input_device_name = selected_device['name']
            
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=channels,
                dtype='float32',
                device=device_index,
                blocksize=self.chunk_size
            ) as stream:
                print("🎙️ System audio stream opened successfully")
                print("💡 Now play some audio (YouTube, music, etc.) to capture it!")
                
                while self.is_recording:
                    try:
                        audio_chunk, overflowed = stream.read(self.chunk_size)
                        if overflowed:
                            print("⚠️ Audio buffer overflow")
                        
                        # Add to buffer
                        self.audio_buffer.append(audio_chunk.copy())
                        
                        # Show progress every few chunks
                        if len(self.audio_buffer) % 10 == 0:
                            total_samples = len(self.audio_buffer) * self.chunk_size
                            print(f"🎵 Audio buffer: {len(self.audio_buffer)} chunks ({total_samples/self.sample_rate:.1f}s)")
                        
                        # Check if we have enough audio for processing
                        total_samples = len(self.audio_buffer) * self.chunk_size
                        if total_samples >= self.buffer_size:
                            print(f"🎵 Audio buffer full! Processing {len(self.audio_buffer)} chunks ({total_samples/self.sample_rate:.1f}s)")
                            
                            # Combine audio chunks
                            combined_audio = np.concatenate(self.audio_buffer, axis=0)
                            
                            # Put in processing queue
                            self.audio_queue.put(combined_audio.copy())
                            print(f"📤 Added audio to processing queue. Queue size: {self.audio_queue.qsize()}")
                            
                            # Keep overlap for smooth transitions
                            overlap_samples = int(self.buffer_size * self.overlap_ratio)
                            overlap_frames = max(1, overlap_samples // self.chunk_size)
                            self.audio_buffer = self.audio_buffer[-overlap_frames:]
                            print(f"🔄 Kept {overlap_frames} overlap frames")
                            
                    except Exception as e:
                        print(f"❌ Error reading system audio: {e}")
                        break
                        
        except Exception as e:
            print(f"❌ Error opening system audio stream: {e}")
            print("💡 Make sure your audio routing is set up correctly (BlackHole, etc.)")
    
    def _process_audio(self):
        """Process audio chunks and generate subtitles"""
        print("🔄 Audio processing thread started")
        while self.is_recording or not self.audio_queue.empty():
            try:
                # Get audio from queue
                audio_data = self.audio_queue.get(timeout=1)
                
                if audio_data is None:
                    continue
                
                print(f"🎵 Processing audio chunk: {len(audio_data)} samples")
                
                # Process audio
                self._transcribe_and_translate(audio_data)
                
            except queue.Empty:
                if self.is_recording:
                    print("⏳ Waiting for audio data...")
                continue
            except Exception as e:
                print(f"❌ Error processing audio: {e}")
        
        print("🔄 Audio processing thread ended")
    
    def _transcribe_and_translate(self, audio_data):
        """Transcribe audio and translate to target language"""
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                sf.write(temp_file.name, audio_data, self.sample_rate)
                temp_path = temp_file.name
            
            try:
                # Transcribe with Whisper
                print("🧠 Transcribing audio...")
                result = self.whisper_model.transcribe(
                    temp_path,
                    language=self.input_lang if self.input_lang != 'auto' else None,
                    task="transcribe"
                )
                
                transcribed_text = result['text'].strip()
                
                if transcribed_text and transcribed_text != self.last_subtitle:
                    print(f"📝 Transcribed: {transcribed_text}")
                    
                    # Translate text
                    translated_text = self._translate_text(transcribed_text)
                    
                    # Update context
                    self._update_context(transcribed_text)
                    
                    # Send to subtitle server
                    self._send_subtitle(transcribed_text, translated_text)
                    
                    self.last_subtitle = transcribed_text
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            print(f"❌ Error in transcription/translation: {e}")
    
    def _translate_text(self, text):
        """Translate text to target language using working methods like text_translator.py"""
        try:
            if self.input_lang == self.output_lang:
                return text  # No translation needed
            
            if not self.translator:
                print("⚠️ Translator not available, using fallback")
                return self._fallback_translate(text)
            
            # Get language codes
            source_code = self.get_language_code(self.input_lang)
            target_code = self.get_language_code(self.output_lang)
            
            print(f"🌍 Translating: '{text}' from {source_code} to {target_code} using {self.translation_method}")
            print(f"🔍 Input lang: {self.input_lang} → source_code: {source_code}")
            print(f"🔍 Output lang: {self.output_lang} → target_code: {target_code}")
            
            # Use the same approach as working files
            if self.translation_method == "googletrans":
                # Like text_translator.py line 528
                result = self.translator.translate(text, src=source_code, dest=target_code)
                translated_text = result.text
                print(f"✅ Google Translate result: '{translated_text}'")
            elif self.translation_method == "deep_translator":
                # Create translator instance and translate
                from deep_translator import GoogleTranslator
                translator = GoogleTranslator(source=source_code, target=target_code)
                translated_text = translator.translate(text)
                print(f"✅ Deep Translate result: '{translated_text}'")
            else:
                return self._fallback_translate(text)
            
            return translated_text
            
        except Exception as e:
            print(f"❌ Translation error: {e}")
            print("🔄 Falling back to fallback translation method")
            return self._fallback_translate(text)
    
    def _fallback_translate(self, text):
        """Fallback translation for common phrases"""
        try:
            if self.output_lang in ['zh', 'zh-cn', 'zh-tw', 'zh-CN', 'zh-TW']:
                # Basic English to Chinese fallback
                fallback_translations = {
                    'hello': '你好',
                    'hi': '你好',
                    'thank you': '谢谢',
                    'thanks': '谢谢',
                    'goodbye': '再见',
                    'bye': '再见',
                    'yes': '是',
                    'no': '不',
                    'please': '请',
                    'sorry': '对不起',
                    'excuse me': '打扰一下',
                    'how are you': '你好吗',
                    'i love you': '我爱你',
                    'welcome': '欢迎',
                    'good': '好',
                    'bad': '坏',
                    'okay': '好的',
                    'ok': '好的',
                    'water': '水',
                    'food': '食物',
                    'music': '音乐',
                    'video': '视频',
                    'youtube': 'YouTube',
                    'subscribe': '订阅',
                    'like': '喜欢',
                    'comment': '评论'
                }
                
                text_lower = text.lower().strip()
                if text_lower in fallback_translations:
                    translated = fallback_translations[text_lower]
                    print(f"🌍 Fallback translation: {text} → {translated}")
                    return translated
                
                # Try partial matches for longer phrases
                for key, value in fallback_translations.items():
                    if key in text_lower:
                        print(f"🌍 Partial fallback match: {text} → {value}")
                        return f"{value} ({text})"
            
            # If no fallback available, return original text with note
            print(f"⚠️ No fallback translation available for: {text}")
            return f"[Original: {text}] (Translation not available)"
            
        except Exception as e:
            print(f"❌ Fallback translation error: {e}")
            return f"[Original: {text}] (Translation failed)"
    
    def _update_context(self, text):
        """Update context window for better translation"""
        self.context_window.append(text)
        
        # Keep context window manageable
        if len(self.context_window) > self.max_context_length:
            self.context_window = self.context_window[-self.max_context_length:]
    
    def _send_subtitle(self, original_text, translated_text):
        """Send subtitle to the server"""
        try:
            subtitle_data = {
                "original": original_text,
                "translated": translated_text,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "input_lang": self.input_lang,
                "output_lang": self.output_lang
            }
            
            response = requests.post(
                f"{self.server_url}/update",
                json=subtitle_data,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            
            if response.status_code == 200:
                print(f"✅ Subtitle sent to server: {original_text}")
            else:
                print(f"⚠️ Server response: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error sending subtitle to server: {e}")
    
    def _update_server_status(self, is_recording):
        """Update server recording status"""
        try:
            status_data = {
                "recording": is_recording,
                "timestamp": datetime.now().isoformat()
            }
            
            # We'll add a status endpoint later, for now just log
            if is_recording:
                print("🔄 Server status: Recording started")
            else:
                print("🔄 Server status: Recording stopped")
                
        except Exception as e:
            print(f"❌ Error updating server status: {e}")
    
    def get_status(self):
        """Get current processor status"""
        return {
            "is_recording": self.is_recording,
            "input_language": getattr(self, 'input_lang', 'Not set'),
            "output_language": getattr(self, 'output_lang', 'Not set'),
            "whisper_loaded": self.whisper_model is not None,
            "translator_ready": self.translation_method != "none",
            "translation_method": getattr(self, 'translation_method', 'none'),
            "audio_buffer_size": len(self.audio_buffer),
            "queue_size": self.audio_queue.qsize()
        }

def test_audio_processor():
    """Test the audio processor"""
    print("🧪 Testing Audio Processor")
    print("=" * 50)
    
    processor = AudioProcessor()
    
    print("\n📊 Processor Status:")
    status = processor.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\n🎤 Ready to test recording!")
    print("💡 Use processor.start_recording() and processor.stop_recording()")
    
    return processor

if __name__ == "__main__":
    print("🚀 Audio Processor - Step 3")
    print("=" * 50)
    print("This handles audio recording, Whisper transcription, and translation")
    print("=" * 50)
    
    # Test the processor
    processor = test_audio_processor()
    
    # Keep the program running for testing
    try:
        print("\n⏳ Press Ctrl+C to exit...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Audio processor test ended")

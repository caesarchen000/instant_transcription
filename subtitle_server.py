#!/usr/bin/env python3
"""
Working Subtitle Generator Server
Step 6: Integrated with working audio processor
"""

import http.server
import socketserver
import json
import threading
import time
from datetime import datetime
import sys
import os

# Import our working audio processor
try:
    from audio_processor import AudioProcessor
    print("✅ Audio processor imported successfully")
except ImportError as e:
    print(f"❌ Error importing audio processor: {e}")
    print("💡 Make sure audio_processor.py is in the same directory")
    sys.exit(1)

class SubtitleServer:
    def __init__(self, host='localhost', port=8770):
        self.host = host
        self.port = port
        self.latest_subtitle = None
        self.is_recording = False
        self.audio_processor = None
        
        # Initialize audio processor
        self._initialize_audio_processor()
        
    def is_audio_processor_ready(self):
        """Check if audio processor is ready for recording"""
        return (self.audio_processor is not None and 
                self.audio_processor.whisper_model is not None)
        
    def _initialize_audio_processor(self):
        """Initialize the audio processor"""
        try:
            print("🎤 Initializing audio processor...")
            self.audio_processor = AudioProcessor(server_url=f'http://{self.host}:{self.port}')
            
            if self.audio_processor.input_device is not None:
                print("✅ Audio processor ready!")
                print(f"🎯 Input device: {self.audio_processor.input_device}")
            else:
                print("⚠️ Audio processor initialized but no input device found")
                
        except Exception as e:
            print(f"❌ Error initializing audio processor: {e}")
            self.audio_processor = None
        
    def start_recording(self, input_lang='en', output_lang='zh'):
        """Start audio recording"""
        if not self.is_audio_processor_ready():
            print("❌ Audio processor not ready for recording")
            return False
        
        if self.is_recording:
            print("⚠️ Already recording!")
            return False
        
        try:
            print(f"🎙️ Starting recording: {input_lang} → {output_lang}")
            
            # Start recording in audio processor
            success = self.audio_processor.start_recording(input_lang, output_lang)
            
            if success:
                self.is_recording = True
                print("✅ Recording started successfully")
                return True
            else:
                print("❌ Failed to start recording")
                return False
                
        except Exception as e:
            print(f"❌ Error starting recording: {e}")
            return False
    
    def stop_recording(self):
        """Stop audio recording"""
        if not self.is_recording:
            print("⚠️ Not recording!")
            return False
        
        try:
            print("⏹️ Stopping recording...")
            
            # Stop recording in audio processor
            success = self.audio_processor.stop_recording()
            
            if success:
                self.is_recording = False
                print("✅ Recording stopped successfully")
                return True
            else:
                print("❌ Failed to stop recording")
                return False
                
        except Exception as e:
            print(f"❌ Error stopping recording: {e}")
            return False
        
    def update_subtitle(self, original_text, translated_text, timestamp=None):
        """Update the latest subtitle data"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%H:%M:%S")
        
        self.latest_subtitle = {
            "original": original_text,
            "translated": translated_text,
            "timestamp": timestamp,
            "type": "subtitle"
        }
        print(f"📝 Subtitle updated: {original_text} -> {translated_text}")
    
    def set_recording_status(self, is_recording):
        """Set the recording status"""
        self.is_recording = is_recording
        status = "Recording..." if is_recording else "Stopped"
        print(f"🎙️ Recording status: {status}")
    
    def get_latest_subtitle(self):
        """Get the latest subtitle data"""
        return self.latest_subtitle or {
            "original": "No subtitles yet",
            "translated": "还没有字幕",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "type": "subtitle"
        }
    
    def get_status(self):
        """Get server status"""
        return {
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "recording": self.is_recording,
            "audio_processor_ready": self.is_audio_processor_ready(),
            "input_device": getattr(self.audio_processor, 'input_device', None) if self.audio_processor else None,
            "input_device_name": getattr(self.audio_processor, 'input_device_name', None) if self.audio_processor else None,
            "whisper_ready": getattr(self.audio_processor, 'whisper_model', None) is not None if self.audio_processor else False,
            "debug_info": {
                "audio_processor_exists": self.audio_processor is not None,
                "whisper_model_exists": getattr(self.audio_processor, 'whisper_model', None) is not None if self.audio_processor else None,
                "translation_method": getattr(self.audio_processor, 'translation_method', None) if self.audio_processor else None
            }
        }

class SubtitleHTTPHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.subtitle_server = kwargs.pop('subtitle_server', None)
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        print(f"🌐 GET request: {self.path}")
        
        if self.path == '/':
            # Serve the HTML page
            print("📄 Serving HTML page")
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Read and serve the HTML file
            try:
                with open('subtitle_interface.html', 'r', encoding='utf-8') as f:
                    html_content = f.read()
            except FileNotFoundError:
                # Fallback if HTML file doesn't exist
                html_content = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Subtitle Generator - Step 2</title>
                </head>
                <body>
                    <h1>Subtitle Generator - Step 2</h1>
                    <p>HTML interface created! But subtitle_interface.html file is missing.</p>
                    <p>Make sure the HTML file is in the same directory as subtitle_server.py</p>
                </body>
                </html>
                """
            self.wfile.write(html_content.encode('utf-8'))
                
        elif self.path == '/subtitle':
            # Return latest subtitle data
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            subtitle_data = self.subtitle_server.get_latest_subtitle()
            self.wfile.write(json.dumps(subtitle_data, ensure_ascii=False).encode('utf-8'))
            
        elif self.path == '/status':
            # Return server status
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            status_data = self.subtitle_server.get_status()
            self.wfile.write(json.dumps(status_data).encode('utf-8'))
            
        elif self.path == '/test_audio':
            # Test audio devices
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            try:
                import sounddevice as sd
                devices = sd.query_devices()
                input_devices = [d for d in devices if d.get('max_inputs', 0) > 0]
                
                test_data = {
                    "all_devices": [{"name": d.get('name', 'Unknown'), "max_inputs": d.get('max_inputs', 0), "max_outputs": d.get('max_outputs', 0)} for d in devices],
                    "input_devices": [{"name": d.get('name', 'Unknown'), "max_inputs": d.get('max_inputs', 0)} for d in input_devices],
                    "default_input": sd.default.device[0] if hasattr(sd, 'default') else None,
                    "default_output": sd.default.device[1] if hasattr(sd, 'default') else None
                }
                
                self.wfile.write(json.dumps(test_data, indent=2).encode('utf-8'))
                
            except Exception as e:
                error_data = {"error": str(e)}
                self.wfile.write(json.dumps(error_data).encode('utf-8'))
            
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")
    
    def do_POST(self):
        """Handle POST requests for subtitle updates and recording control"""
        if self.path == '/update' or self.path == '/subtitle':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                if 'original' in data and 'translated' in data:
                    self.subtitle_server.update_subtitle(
                        data['original'], 
                        data['translated'], 
                        data.get('timestamp')
                    )
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "success"}).encode('utf-8'))
                else:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Invalid data format")
            except json.JSONDecodeError:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid JSON")
                
        elif self.path == '/start_recording':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                input_lang = data.get('input_lang', 'en')
                output_lang = data.get('output_lang', 'zh')
                
                # Actually start recording using our audio processor
                success = self.subtitle_server.start_recording(input_lang, output_lang)
                
                if success:
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "recording_started", "success": True}).encode('utf-8'))
                else:
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "recording_failed", "success": False}).encode('utf-8'))
                
            except json.JSONDecodeError:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid JSON")
                
        elif self.path == '/stop_recording':
            # Actually stop recording using our audio processor
            success = self.subtitle_server.stop_recording()
            
            if success:
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "recording_stopped", "success": True}).encode('utf-8'))
            else:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "recording_failed", "success": False}).encode('utf-8'))
            
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")
    
    def log_message(self, format, *args):
        """Custom logging"""
        print(f"🌐 HTTP: {format % args}")

def start_server():
    """Start the subtitle server"""
    subtitle_server = SubtitleServer()
    
    # Create custom handler with subtitle server reference
    class CustomHandler(SubtitleHTTPHandler):
        def __init__(self, *args, **kwargs):
            kwargs['subtitle_server'] = subtitle_server
            super().__init__(*args, **kwargs)
    
    # Start server
    with socketserver.TCPServer(("", subtitle_server.port), CustomHandler) as httpd:
        print(f"🌐 Subtitle server started on http://{subtitle_server.host}:{subtitle_server.port}")
        print(f"📱 Open http://localhost:{subtitle_server.port} in your browser")
        print(f"📝 Subtitles available at http://localhost:{subtitle_server.port}/subtitle")
        print(f"🔗 API endpoints:")
        print(f"   POST /update - Send subtitle data")
        print(f"   GET /subtitle - Get latest subtitle")
        print(f"   GET /status - Get server status")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")

if __name__ == "__main__":
    print("🚀 Starting Working Subtitle Generator - Step 6")
    print("=" * 50)
    print("This runs the complete system: HTML + Working Audio Processor")
    print("=" * 50)
    start_server()

#!/usr/bin/env python3
"""
Picture-in-Picture Subtitle Overlay
Floating subtitle window that overlays on top of other applications
"""

import tkinter as tk
from tkinter import ttk
import json
import requests
import threading
import time
from datetime import datetime

class PiPSubtitleOverlay:
    def __init__(self, server_url='http://localhost:8766'):
        self.server_url = server_url
        self.is_visible = False
        self.is_dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        
        # Create the floating window
        self.create_pip_window()
        
        # Start subtitle polling
        self.start_subtitle_polling()
    
    def create_pip_window(self):
        """Create the floating PiP subtitle window"""
        self.root = tk.Tk()
        self.root.title("PiP Subtitles")
        
        # Make window floating and always on top
        self.root.attributes('-topmost', True)
        self.root.overrideredirect(True)  # Remove window decorations
        
        # Set window size and position
        self.root.geometry("400x150")
        self.root.geometry("+100+100")  # Position at top-left
        
        # Make window semi-transparent
        self.root.attributes('-alpha', 0.9)
        
        # Create main frame with dark theme
        self.main_frame = tk.Frame(self.root, bg='#1a1a1a', relief='raised', bd=2)
        self.main_frame.pack(fill='both', expand=True, padx=2, pady=2)
        
        # Title bar (draggable)
        self.title_bar = tk.Frame(self.main_frame, bg='#2d2d2d', height=30)
        self.title_bar.pack(fill='x')
        self.title_bar.pack_propagate(False)
        
        # Title label
        self.title_label = tk.Label(
            self.title_bar, 
            text="🎬 Live Subtitles", 
            bg='#2d2d2d', 
            fg='#00d4ff',
            font=('Arial', 10, 'bold')
        )
        self.title_label.pack(side='left', padx=10, pady=5)
        
        # Control buttons
        self.controls_frame = tk.Frame(self.title_bar, bg='#2d2d2d')
        self.controls_frame.pack(side='right', padx=5)
        
        # Minimize button
        self.minimize_btn = tk.Button(
            self.controls_frame,
            text="−",
            bg='#ff6b6b',
            fg='white',
            font=('Arial', 8, 'bold'),
            width=2,
            command=self.toggle_visibility
        )
        self.minimize_btn.pack(side='left', padx=2)
        
        # Close button
        self.close_btn = tk.Button(
            self.controls_frame,
            text="×",
            bg='#ff6b6b',
            fg='white',
            font=('Arial', 8, 'bold'),
            width=2,
            command=self.root.quit
        )
        self.close_btn.pack(side='left', padx=2)
        
        # Subtitle content area
        self.content_frame = tk.Frame(self.main_frame, bg='#1a1a1a')
        self.content_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Original text
        self.original_label = tk.Label(
            self.content_frame,
            text="Original Text",
            bg='#1a1a1a',
            fg='#ffffff',
            font=('Arial', 12),
            wraplength=350,
            justify='center'
        )
        self.original_label.pack(pady=(0, 5))
        
        # Translated text
        self.translated_label = tk.Label(
            self.content_frame,
            text="Translated Text",
            bg='#1a1a1a',
            fg='#00d4ff',
            font=('Arial', 14, 'bold'),
            wraplength=350,
            justify='center'
        )
        self.translated_label.pack(pady=(0, 5))
        
        # Timestamp
        self.timestamp_label = tk.Label(
            self.content_frame,
            text="00:00:00",
            bg='#1a1a1a',
            fg='#888888',
            font=('Arial', 10)
        )
        self.timestamp_label.pack()
        
        # Bind mouse events for dragging
        self.title_bar.bind('<Button-1>', self.start_drag)
        self.title_bar.bind('<B1-Motion>', self.drag)
        self.title_bar.bind('<ButtonRelease-1>', self.stop_drag)
        
        # Bind double-click to toggle visibility
        self.title_bar.bind('<Double-Button-1>', self.toggle_visibility)
        
        # Status indicator
        self.status_label = tk.Label(
            self.main_frame,
            text="● Connected",
            bg='#1a1a1a',
            fg='#00ff00',
            font=('Arial', 8)
        )
        self.status_label.pack(side='bottom', pady=2)
    
    def start_drag(self, event):
        """Start dragging the window"""
        self.is_dragging = True
        self.drag_start_x = event.x
        self.drag_start_y = event.y
    
    def drag(self, event):
        """Drag the window"""
        if self.is_dragging:
            x = self.root.winfo_x() + (event.x - self.drag_start_x)
            y = self.root.winfo_y() + (event.y - self.drag_start_y)
            self.root.geometry(f"+{x}+{y}")
    
    def stop_drag(self, event):
        """Stop dragging the window"""
        self.is_dragging = False
    
    def toggle_visibility(self):
        """Toggle window visibility"""
        if self.is_visible:
            self.root.withdraw()
            self.is_visible = False
            self.minimize_btn.config(text="□")
        else:
            self.root.deiconify()
            self.is_visible = True
            self.minimize_btn.config(text="−")
    
    def update_subtitles(self, subtitle_data):
        """Update the subtitle display"""
        try:
            if subtitle_data and 'original' in subtitle_data:
                # Update original text
                original_text = subtitle_data.get('original', 'No subtitles yet')
                if original_text != 'No subtitles yet':
                    self.original_label.config(text=original_text)
                
                # Update translated text
                translated_text = subtitle_data.get('translated', '')
                if translated_text and translated_text != original_text:
                    self.translated_label.config(text=translated_text)
                else:
                    self.translated_label.config(text=original_text)
                
                # Update timestamp
                timestamp = subtitle_data.get('timestamp', '')
                if timestamp:
                    self.timestamp_label.config(text=timestamp)
                
                # Update status
                self.status_label.config(text="● Live", fg='#00ff00')
                
        except Exception as e:
            print(f"❌ Error updating subtitles: {e}")
    
    def fetch_subtitles(self):
        """Fetch latest subtitles from server"""
        try:
            response = requests.get(f"{self.server_url}/subtitle", timeout=2)
            if response.status_code == 200:
                subtitle_data = response.json()
                self.update_subtitles(subtitle_data)
            else:
                self.status_label.config(text="● Error", fg='#ff0000')
                
        except requests.exceptions.RequestException:
            self.status_label.config(text="● Disconnected", fg='#ff6b6b')
        except Exception as e:
            print(f"❌ Error fetching subtitles: {e}")
    
    def start_subtitle_polling(self):
        """Start polling for subtitle updates"""
        def poll_subtitles():
            while True:
                try:
                    self.fetch_subtitles()
                    time.sleep(0.5)  # Poll every 500ms
                except Exception as e:
                    print(f"❌ Polling error: {e}")
                    time.sleep(1)
        
        # Start polling in background thread
        self.polling_thread = threading.Thread(target=poll_subtitles, daemon=True)
        self.polling_thread.start()
    
    def run(self):
        """Run the PiP overlay"""
        try:
            print("🎬 PiP Subtitle Overlay started!")
            print(f"🌐 Connecting to server: {self.server_url}")
            print("💡 Drag the title bar to move the window")
            print("💡 Double-click title bar to minimize/maximize")
            print("💡 Click × to close")
            
            self.root.mainloop()
            
        except KeyboardInterrupt:
            print("\n👋 PiP overlay stopped")
        except Exception as e:
            print(f"❌ Error running PiP overlay: {e}")

def main():
    """Main function"""
    print("🚀 Starting Picture-in-Picture Subtitle Overlay")
    print("=" * 50)
    
    # Create and run the PiP overlay
    pip_overlay = PiPSubtitleOverlay()
    pip_overlay.run()

if __name__ == "__main__":
    main()

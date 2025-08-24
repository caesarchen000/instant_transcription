#!/usr/bin/env python3
"""
Test BlackHole Audio Routing
This script tests if audio is properly flowing through BlackHole
"""

import sounddevice as sd
import numpy as np
import time

def test_blackhole_audio():
    """Test if we can capture audio from BlackHole 16ch"""
    print("🎤 Testing BlackHole 16ch audio capture...")
    
    # Get device info
    devices = sd.query_devices()
    print(f"📱 Found {len(devices)} audio devices")
    
    # Find BlackHole 16ch
    blackhole_device = None
    for i, device in enumerate(devices):
        if 'BlackHole 16ch' in device['name']:
            blackhole_device = device
            blackhole_index = i
            print(f"✅ Found BlackHole 16ch at index {i}")
            print(f"   Input channels: {device.get('max_input_channels', 'N/A')}")
            print(f"   Output channels: {device.get('max_output_channels', 'N/A')}")
            break
    
    if not blackhole_device:
        print("❌ BlackHole 16ch not found!")
        return False
    
    # Test audio capture
    print(f"\n🎙️ Testing audio capture from BlackHole 16ch...")
    print("=" * 60)
    print("🚨 IMPORTANT: Before continuing, you MUST:")
    print("   1. Open System Preferences → Sound")
    print("   2. Set 'BlackHole 16ch' as your OUTPUT device")
    print("   3. Play some audio (YouTube, music, etc.)")
    print("   4. Make sure you can hear the audio")
    print("=" * 60)
    
    input("Press Enter when you've set BlackHole 16ch as output and are playing audio...")
    
    try:
        with sd.InputStream(
            device=blackhole_index,
            channels=2,
            samplerate=44100,
            dtype='float32',
            blocksize=1024
        ) as stream:
            print("🎵 Audio stream opened successfully!")
            print("🎧 Listening for audio... (10 seconds)")
            
            # Listen for 10 seconds
            start_time = time.time()
            audio_detected = False
            max_level = 0
            
            while time.time() - start_time < 10:
                try:
                    audio_chunk, overflowed = stream.read(1024)
                    if overflowed:
                        print("⚠️ Audio buffer overflow")
                    
                    # Check audio level
                    audio_level = np.max(np.abs(audio_chunk))
                    max_level = max(max_level, audio_level)
                    
                    if audio_level > 0.001:
                        if not audio_detected:
                            print(f"🎵 Audio detected! Level: {audio_level:.4f}")
                            audio_detected = True
                        print(f"🎵 Audio level: {audio_level:.4f}")
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"❌ Error reading audio: {e}")
                    break
            
            print(f"\n📊 Test Results:")
            print(f"   Max audio level detected: {max_level:.6f}")
            print(f"   Audio detected: {'✅ YES' if audio_detected else '❌ NO'}")
            
            if audio_detected:
                print("\n✅ SUCCESS: Audio is flowing through BlackHole!")
                print("🎯 Your audio routing is working correctly")
                print("🚀 You can now use the subtitle generator!")
                return True
            else:
                print("\n❌ No audio detected from BlackHole")
                print("💡 Troubleshooting steps:")
                print("   1. Make sure BlackHole 16ch is set as OUTPUT in System Preferences")
                print("   2. Play some audio (YouTube, music, etc.)")
                print("   3. Check that you can hear the audio")
                print("   4. Try restarting your audio app")
                return False
                
    except Exception as e:
        print(f"❌ Error opening audio stream: {e}")
        return False

if __name__ == "__main__":
    print("🔍 BlackHole Audio Routing Test")
    print("=" * 40)
    test_blackhole_audio()

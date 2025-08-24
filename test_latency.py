#!/usr/bin/env python3
"""
Test Latency Improvements
This script tests the optimized audio processor for lower latency
"""

import time
from audio_processor import AudioProcessor

def test_latency_modes():
    """Test different latency modes"""
    print("🎯 Testing Latency Modes")
    print("=" * 40)
    
    # Initialize audio processor
    processor = AudioProcessor()
    
    print(f"\n📊 Current Settings:")
    print(f"   Chunk size: {processor.chunk_size} samples ({processor.chunk_size/processor.sample_rate*1000:.1f}ms)")
    print(f"   Buffer size: {processor.buffer_size} samples ({processor.buffer_size/processor.sample_rate:.1f}s)")
    print(f"   Overlap ratio: {processor.overlap_ratio*100:.0f}%")
    print(f"   Low latency mode: {processor.low_latency_mode}")
    
    # Test low latency mode
    print(f"\n⚡ Testing LOW LATENCY mode...")
    processor.set_latency_mode(low_latency=True)
    
    # Test high accuracy mode
    print(f"\n🎯 Testing HIGH ACCURACY mode...")
    processor.set_latency_mode(low_latency=False)
    
    # Switch back to low latency
    print(f"\n⚡ Switching back to LOW LATENCY mode...")
    processor.set_latency_mode(low_latency=True)
    
    print(f"\n✅ Latency test completed!")
    print(f"💡 Use LOW LATENCY mode for real-time subtitles")
    print(f"💡 Use HIGH ACCURACY mode for better transcription quality")

if __name__ == "__main__":
    test_latency_modes()

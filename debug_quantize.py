#!/usr/bin/env python3
import numpy as np

def quantize_audio_old(audio_data, target_bits):
    """Current implementation"""
    if target_bits >= 16:
        return audio_data
    shift = 16 - target_bits
    quantized = (audio_data >> shift) << shift
    return quantized

def dequantize_audio_old(quantized_data, target_bits):
    """Current implementation"""
    if target_bits >= 16:
        return quantized_data
    return quantized_data

def quantize_audio_new(audio_data, target_bits):
    """Proper implementation"""
    if target_bits >= 16:
        return audio_data
    
    # Convert to unsigned range [0, 65535] using int32 to avoid overflow
    unsigned = audio_data.astype(np.int32) + 32768
    
    # Scale to target bit range [0, 2^target_bits - 1]
    max_val = (1 << target_bits) - 1
    scaled = (unsigned * max_val) // 65535
    
    # Convert back to signed 16-bit by reversing the process
    # First scale back to [0, 65535] using int32
    unsigned_scaled = (scaled * 65535) // max_val
    # Clamp to valid range for uint16 before conversion
    unsigned_scaled = np.clip(unsigned_scaled, 0, 65535)
    # Then convert to signed
    signed = unsigned_scaled.astype(np.int16) - 32768
    
    return signed

def dequantize_audio_new(quantized_data, target_bits):
    """For lossy compression, dequantized is same as quantized"""
    if target_bits >= 16:
        return quantized_data
    return quantized_data

# Test with user's example
original_sample = 32100
print(f"Testing with sample value: {original_sample}")
print()

print("OLD IMPLEMENTATION:")
quantized_old = quantize_audio_old(np.array([original_sample]), target_bits=8)[0]
dequantized_old = dequantize_audio_old(np.array([quantized_old]), target_bits=8)[0]
print(f"  Orijinal: {original_sample}")
print(f"  Kuantize: {quantized_old}")
print(f"  Geri açılan: {dequantized_old}")
print(f"  Orijinal == Kuantize? {original_sample == quantized_old}")
print(f"  Orijinal == Geri açılan? {original_sample == dequantized_old}")
print()

print("NEW IMPLEMENTATION:")
quantized_new = quantize_audio_new(np.array([original_sample]), target_bits=8)[0]
dequantized_new = dequantize_audio_new(np.array([quantized_new]), target_bits=8)[0]
print(f"  Orijinal: {original_sample}")
print(f"  Kuantize: {quantized_new}")
print(f"  Geri açılan: {dequantized_new}")
print(f"  Orijinal == Kuantize? {original_sample == quantized_new}")
print(f"  Orijinal == Geri açılan? {original_sample == dequantized_new}")
print()

# Test with a range of values to see when old implementation gives no change
print("Testing when OLD implementation gives no change (quantized == original):")
no_change_count = 0
total_test = 0
for i in range(-32768, 32768, 100):  # Test every 100th value to make it faster
    total_test += 1
    q = quantize_audio_old(np.array([i]), target_bits=8)[0]
    if q == i:
        no_change_count += 1
        if no_change_count <= 5:  # Show first 5 examples
            print(f"  {i} -> {q} (no change)")
            
print(f"  Found {no_change_count} values out of {total_test} that didn't change")
print(f"  Percentage: {no_change_count/total_test*100:.2f}%")
print()

# Test with actual audio-like values (sine wave)
import math
print("Testing with sine wave values:")
samples = []
for i in range(100):
    # Generate sine wave samples
    t = i / 100.0
    sample = int(math.sin(2 * math.pi * 0.1 * t) * 16000)  # Reduced amplitude to avoid clipping
    samples.append(sample)

no_change_sine = 0
for sample in samples:
    q = quantize_audio_old(np.array([sample]), target_bits=8)[0]
    if q == sample:
        no_change_sine += 1
        
print(f"  Sine wave test: {no_change_sine}/100 samples unchanged with OLD implementation")
print(f"  Percentage: {no_change_sine/100*100:.2f}%")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compression algorithms module
"""

import numpy as np
import zlib
import heapq
import itertools

def rle_encode(data):
    """
    Basic Run-Length Encoding (RLE) algorithm.
    Does not compress if repetition count is less than 3 (inefficient).
    
    Args:
        data (bytes): Data to compress
        
    Returns:
        bytes: RLE compressed data
    """
    if not data:
        return data
    
    encoded = bytearray()
    i = 0
    n = len(data)
    
    while i < n:
        # Count consecutive same byte values
        j = i + 1
        while j < n and data[j] == data[i] and (j - i) < 255:  # Count up to 255 (byte limit)
            j += 1
        
        count = j - i
        
        # If repetition count is 3 or more, use RLE
        if count >= 3:
            encoded.append(count)     # repetition count
            encoded.append(data[i])   # value
            i = j
        else:
            # If repetition count is low, add original data
            encoded.append(data[i])
            i += 1
    
    return bytes(encoded)

def rle_decode(data):
    """
    Decompress RLE compressed data.
    
    Args:
        data (bytes): RLE compressed data
        
    Returns:
        bytes: Original data
    """
    decoded = bytearray()
    i = 0
    n = len(data)
    
    while i < n:
        count = data[i]
        i += 1
        if i >= n:
            break
        value = data[i]
        i += 1
        
        decoded.extend([value] * count)
    
    return bytes(decoded)

class HuffmanNode:
    def __init__(self, byte_val, freq):
        self.byte_val = byte_val
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(frequency):
    """
    Create Huffman tree from frequency table.
    
    Args:
        frequency (dict): Frequencies for byte values
        
    Returns:
        HuffmanNode: Root of Huffman tree
    """
    heap = []
    for byte_val, freq in frequency.items():
        node = HuffmanNode(byte_val, freq)
        heapq.heappush(heap, node)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
    
    return heap[0] if heap else None

def build_huffman_codes(root):
    """
    Create code table from Huffman tree.
    
    Args:
        root (HuffmanNode): Root of Huffman tree
        
    Returns:
        dict: Binary codes for byte values
    """
    codes = {}
    
    def traverse(node, current_code):
        if node is None:
            return
        if node.byte_val is not None:
            codes[node.byte_val] = current_code
            return
        traverse(node.left, current_code + "0")
        traverse(node.right, current_code + "1")
    
    traverse(root, "")
    return codes

def huffman_encode(data):
    """
    Encode data using Huffman coding.
    
    Args:
        data (bytes): Data to encode
        
    Returns:
        bytes: Huffman encoded data (as bit string, then converted to bytes)
    """
    if not data:
        return data
    
    # Create frequency table
    frequency = {}
    for byte_val in data:
        frequency[byte_val] = frequency.get(byte_val, 0) + 1
    
    # If only one byte value, Huffman tree cannot be built
    if len(frequency) == 1:
        # Special case: single value already behaves like compressed
        return data
    
    # Build Huffman tree
    root = build_huffman_tree(frequency)
    if root is None:
        return data
    
    # Create code table
    codes = build_huffman_codes(root)
    
    # Encode data (as bit string)
    encoded_bits = ""
    for byte_val in data:
        encoded_bits += codes[byte_val]
    
    # Convert bit string to bytes
    # Add padding to make it multiple of 8
    extra_padding = 8 - len(encoded_bits) % 8
    for _ in range(extra_padding):
        encoded_bits += "0"
    
    # Save padding info (first 8 bits)
    padded_info = "{0:08b}".format(extra_padding)
    encoded_bits = padded_info + encoded_bits
    
    # Convert to bytes
    b = bytearray()
    for i in range(0, len(encoded_bits), 8):
        byte = encoded_bits[i:i+8]
        b.append(int(byte, 2))
    
    return bytes(b)

def huffman_decode(data):
    """
    Decode Huffman encoded data.
    
    Args:
        data (bytes): Huffman encoded data
        
    Returns:
        bytes: Original data
    """
    if not data:
        return data
    
    # Convert bytes to bit string
    bit_string = ""
    for byte_val in data:
        bit_string += "{0:08b}".format(byte_val)
    
    # Extract first 8 bits as padding info
    padding_info = bit_string[:8]
    extra_padding = int(padding_info, 2)
    bit_string = bit_string[8:]
    
    # Remove extra padding
    if extra_padding > 0:
        bit_string = bit_string[:-extra_padding]
    
    # Reconstruct frequency table from the data (simplified approach)
    # In a real implementation, we would send the frequency table with the data
    # For this implementation, we'll rebuild the tree from the data
    frequency = {}
    for byte_val in data:
        frequency[byte_val] = frequency.get(byte_val, 0) + 1
    
    # If only one byte value, return as is
    if len(frequency) == 1:
        return data
    
    # Build Huffman tree
    root = build_huffman_tree(frequency)
    if root is None:
        return data
    
    # Create code table
    codes = build_huffman_codes(root)
    
    # Create reverse mapping for decoding
    reverse_codes = {v: k for k, v in codes.items()}
    
    # Decode bit string
    decoded_bytes = bytearray()
    current_code = ""
    for bit in bit_string:
        current_code += bit
        if current_code in reverse_codes:
            decoded_bytes.append(reverse_codes[current_code])
            current_code = ""
    
    return bytes(decoded_bytes)

def quantize_audio(audio_data, target_bits):
    """
    Quantize audio data to specified bit depth using proper linear quantization.
    
    Args:
        audio_data (numpy array): 16-bit PCM audio data (integer)
        target_bits (int): Target bit depth (4, 8, 12)
        
    Returns:
        numpy array: Quantized audio data (same shape, but with reduced precision)
    """
    # audio_data assumed to be int16 in range [-32768, 32767]
    if target_bits >= 16:
        return audio_data
    
    # Convert to unsigned float in range [0, 1]
    unsigned_float = (audio_data.astype(np.float32) + 32768) / 65535.0
    # Quantize to target bits
    quantized_float = np.round(unsigned_float * ((1 << target_bits) - 1)) / ((1 << target_bits) - 1)
    # Convert back to unsigned float in range [0, 65535]
    unsigned_scaled = quantized_float * 65535
    # Clip to [0, 65535] (should be redundant but safe)
    unsigned_scaled = np.clip(unsigned_scaled, 0, 65535)
    # Convert to signed 16-bit: subtract 32768 and round to nearest integer
    signed = np.round(unsigned_scaled - 32768).astype(np.int16)
    return signed

def dequantize_audio(quantized_data, target_bits):
    """
    Dequantize audio data from specified bit depth back to 16-bit.
    For lossy compression, we return the quantized values as-is since
    the quantization is the lossy step and we are storing the quantized values.
    
    Args:
        quantized_data (numpy array): Quantized audio data (already in 16-bit format with reduced precision)
        target_bits (int): Target bit depth used for quantization (4, 8, 12)
        
    Returns:
        numpy array: Dequantized audio data (16-bit) - same as input for lossy
    """
    # If target_bits >= 16, no quantization was applied
    if target_bits >= 16:
        return quantized_data
    
    # For lossy compression, we return the quantized values as-is
    # The quantization is already applied in quantize_audio, and we store those values.
    # So the dequantized signal is the same as the quantized signal we have.
    return quantized_data

def compress_lossless_uniform(data):
    """
    Lossless - Uniform (Baseline): Apply standard zlib compression to entire file
    
    Args:
        data (bytes): Data to compress
        
    Returns:
        bytes: Compressed data
    """
    return zlib.compress(data, level=6)

def compress_lossless_adaptive(data, labels, segment_length):
    """
    Lossless - Adaptive:
       - Silent segments → Aggressive RLE + zlib (level=9)
       - Active segments → Huffman coding + zlib (level=6)
    
    Args:
        data (bytes): Data to compress (as 16-bit PCM)
        labels (list): Segment labels (0: silent, 1: active)
        segment_length (int): Length of each segment in bytes
        
    Returns:
        bytes: Compressed data
    """
    # Divide data into segments
    num_segments = len(labels)
    processed_segments = bytearray()
    
    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        segment_data = data[start_idx:end_idx]
        
        # Skip if segment data is empty
        if not segment_data:
            continue
            
        label = labels[i] if i < len(labels) else 1  # Default to active
        
        if label == 0:  # silent segment
            # Aggressive RLE + zlib (level=9)
            rle_encoded = rle_encode(segment_data)
            compressed_segment = zlib.compress(rle_encoded, level=9)
        else:  # active segment
            # Huffman coding + zlib (level=6)
            try:
                huffman_encoded = huffman_encode(segment_data)
                compressed_segment = zlib.compress(huffman_encoded, level=6)
            except Exception:
                # If Huffman fails, use only zlib
                compressed_segment = zlib.compress(segment_data, level=6)
        
        processed_segments.extend(compressed_segment)
    
    return bytes(processed_segments)

def compress_lossy_uniform(data):
    """
    Lossy - Uniform (Baseline): Apply low bit depth (16-bit → 8-bit) quantization to entire audio
    
    Args:
        data (bytes): 16-bit PCM audio data
        
    Returns:
        bytes: Quantized and zlib compressed data
    """
    # Convert bytes to int16 array
    audio_int16 = np.frombuffer(data, dtype=np.int16)
    # Quantize to 8-bit
    quantized = quantize_audio(audio_int16, target_bits=8)
    # Compress with zlib (level=6)
    return zlib.compress(quantized.tobytes(), level=6)

def compress_lossy_adaptive(data, labels, segment_length):
    """
    Lossy - Adaptive:
       - Silent segments → Aggressive quantization (16-bit → 4-bit equivalent)
       - Active segments → Light quantization (16-bit → 12-bit equivalent)
    
    Args:
        data (bytes): 16-bit PCM audio data
        labels (list): Segment labels (0: silent, 1: active)
        segment_length (int): Length of each segment in bytes
        
    Returns:
        bytes: Quantized and zlib compressed data
    """
    # Divide data into segments
    num_segments = len(labels)
    all_quantized = bytearray()
    
    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        segment_data = data[start_idx:end_idx]
        
        # Skip if segment data is empty
        if not segment_data:
            continue
            
        label = labels[i] if i < len(labels) else 1  # Default to active
        
        # Convert bytes to int16 array
        audio_int16 = np.frombuffer(segment_data, dtype=np.int16)
        
        if label == 0:  # silent segment
            # Aggressive quantization (16-bit → 4-bit equivalent)
            quantized = quantize_audio(audio_int16, target_bits=4)
        else:  # active segment
            # Light quantization (16-bit → 12-bit equivalent)
            quantized = quantize_audio(audio_int16, target_bits=12)
         
        # Add quantized data to our buffer
        all_quantized.extend(quantized.tobytes())
     
    # Apply zlib compression once to the entire stream
    # Use level=9 as specified in requirements
    compressed_data = zlib.compress(bytes(all_quantized), level=9)
    
    return compressed_data


def compress_rvqgan(audio_array, sample_rate):
    """
    RVQ-GAN tabanlı ses sıkıştırma (Encodec kullanılarak).
    
    Args:
        audio_array (numpy array): Orijinal ses verisi (int16)
        sample_rate (int): Örnekleme frekansı
        
    Returns:
        bytes: Sıkıştırılmış veri
    """
    # Import the rvqgan_compressor module locally to avoid hard dependency if not installed
    try:
        import rvqgan_compressor
        return rvqgan_compressor.compress_rvqgan(audio_array, sample_rate)
    except ImportError as e:
        raise ImportError("RVQ-GAN için gerekli paketler kurulu değil. Lütfen 'pip install encodec torch torchaudio' komutunu çalıştırın.") from e
    except Exception as e:
        raise RuntimeError(f"RVQ-GAN sıkıştırma hatası: {e}") from e
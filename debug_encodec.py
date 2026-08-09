#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to test EnCodec with our test audio
"""

import numpy as np
import scipy.io.wavfile as wav
import torch
import torchaudio
from encodec.model import EncodecModel

def main():
    # Load test audio
    sample_rate, audio_data = wav.read('test.wav')
    print(f"Loaded audio: sample_rate={sample_rate}, shape={audio_data.shape}, dtype={audio_data.dtype}")
    
    # Convert to mono float32 as in our pipeline
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        audio_data = np.mean(audio_data, axis=1)
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    elif audio_data.dtype == np.uint8:
        audio_data = (audio_data.astype(np.float32) - 128) / 128.0
    else:
        audio_data = audio_data.astype(np.float32)
    
    print(f"After conversion: shape={audio_data.shape}, dtype={audio_data.dtype}")
    print(f"Value range: [{np.min(audio_data):.6f}, {np.max(audio_data):.6f}]")
    
    # Load EnCodec model
    print("Loading EnCodec model...")
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)  # 6 kbps
    print("Model loaded.")
    
    # Prepare audio as in our codec
    # Ensure mono
    if audio_data.ndim == 1:
        audio_data = audio_data.reshape(1, -1)
    elif audio_data.ndim == 2 and audio_data.shape[0] > 1:
        audio_data = audio_data[0:1, :]
    
    # Convert to torch tensor
    audio_tensor = torch.from_numpy(audio_data).float()
    
    # Resample if necessary
    target_sample_rate = 24000
    if sample_rate != target_sample_rate:
        print(f"Resampling from {sample_rate} to {target_sample_rate}")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        audio_tensor = resampler(audio_tensor)
    else:
        print(f"No resampling needed, sample_rate={sample_rate}")
    
    # Normalize to [-1, 1] (should already be, but ensure)
    audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)
    
    # EnCodec expects a batch dimension
    audio_tensor = audio_tensor.unsqueeze(0)  # (1, channels, samples)
    print(f"Input tensor shape: {audio_tensor.shape}")
    
    # Try to encode
    print("Attempting to encode...")
    try:
        encoded_frames = model.encode(audio_tensor)
        print(f"Encoded frames type: {type(encoded_frames)}")
        print(f"Number of frames: {len(encoded_frames)}")
        
        if len(encoded_frames) > 0:
            print(f"First frame type: {type(encoded_frames[0])}")
            if isinstance(encoded_frames[0], tuple) and len(encoded_frames[0]) == 2:
                codes, scale = encoded_frames[0]
                print(f"First frame codes shape: {codes.shape}")
                print(f"First frame scale: {scale}")  # Print the scale object to see what it is
                if scale is not None:
                    print(f"First frame scale shape: {scale.shape}")
                    print(f"Scale dtype: {scale.dtype}")
                    print(f"Scale sample values: {scale[0, :, :10]}")  # First few values
                else:
                    print("Scale is None!")
                print(f"Codes dtype: {codes.dtype}")
                print(f"Codes sample values: {codes[0, :, :10]}")  # First few values
            else:
                print(f"First frame content: {encoded_frames[0]}")
        else:
            print("No frames produced!")
            
    except Exception as e:
        print(f"Error during encoding: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
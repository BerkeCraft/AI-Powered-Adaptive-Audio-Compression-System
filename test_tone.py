#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a simple test wav file for testing the compression system.
"""

import numpy as np
import scipy.io.wavfile as wav

# Parameters
duration = 1.0  # seconds
sample_rate = 44100  # Hz
frequency = 440.0  # Hz (A4 note)

# Generate time array
t = np.linspace(0, duration, int(sample_rate * duration), False)

# Generate sine wave
note = np.sin(frequency * 2 * np.pi * t)

# Normalize to 16-bit range and convert to int16
audio_data = (note * 32767).astype(np.int16)

# Write to wav file
wav.write('test.wav', sample_rate, audio_data)

print("Test wav file 'test.wav' created.")
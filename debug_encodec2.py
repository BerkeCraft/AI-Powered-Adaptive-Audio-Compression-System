#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to inspect EnCodec model attributes
"""

import numpy as np
import scipy.io.wavfile as wav
import torch
import torchaudio
from encodec.model import EncodecModel

def main():
    # Load EnCodec model
    print("Loading EnCodec model...")
    model = EncodecModel.encodec_model_24khz()
    print("Model loaded.")
    
    # Print model attributes
    print("\nModel attributes:")
    for attr in sorted(dir(model)):
        if not attr.startswith('_'):
            print(f"  {attr}: {getattr(model, attr)}")
    
    # Check if it has target_bandwidth or bandwidth
    print("\nChecking for bandwidth attributes:")
    if hasattr(model, 'target_bandwidth'):
        print(f"  target_bandwidth: {model.target_bandwidth}")
    else:
        print("  No target_bandwidth attribute")
    if hasattr(model, 'bandwidth'):
        print(f"  bandwidth: {model.bandwidth}")
    else:
        print("  No bandwidth attribute")
        
    # Check if there is a method to set bandwidth
    print("\nChecking for setter methods:")
    for attr in ['set_target_bandwidth', 'set_bandwidth']:
        if hasattr(model, attr):
            print(f"  Has {attr}")
        else:
            print(f"  No {attr}")

if __name__ == "__main__":
    main()
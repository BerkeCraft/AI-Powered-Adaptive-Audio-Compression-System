#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yardımcı fonksiyonlar modülü
"""

import numpy as np
import zlib

def calculate_snr(original, compressed):
    """
    Signal-to-Noise Ratio (SNR) hesaplar.
    
    Args:
        original (numpy array): Orijinal sinyal
        compressed (numpy array): Sıkıştırılmış/kuantize edilmiş sinyal
        
    Returns:
        float: SNR değeri (dB)
    """
    # Sinyal ve ruido gücünü hesapla
    signal_power = np.mean(original.astype(np.float64) ** 2)
    noise_power = np.mean((original.astype(np.float64) - compressed.astype(np.float64)) ** 2)
    
    if noise_power == 0:
        return float('inf')  # Perfect reconstruction
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def format_snr(snr):
    """
    SNR değerini okunabilir string formatında döndürür.
    
    Args:
        snr (float or None): SNR değeri
        
    Returns:
        str: Formatlanmış SNR stringi
    """
    if snr is None:
        return "-"
    elif np.isinf(snr):
        return "∞"  # Sonsuz
    else:
        return f"{snr:.2f}"

def format_size(size_bytes):
    """
    Bayt boyutunu okunabilir formatta döndürür.
    
    Args:
        size_bytes (int): Bayt cinsinden boyut
        
    Returns:
        str: Formatlanmış boyut (örn. "1.23 MB")
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024.0 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}"

def calculate_compression_ratio(original_size, compressed_size):
    """
    Sıkıştırma oranını hesaplar.
    
    Args:
        original_size (int): Orijinal boyut (bytes)
        compressed_size (int): Sıkıştırılmış boyut (bytes)
        
    Returns:
        float: Sıkıştırma oranı (0-1 arası, 1 = %100 kazanç)
    """
    if original_size == 0:
        return 0.0
    return 1.0 - (compressed_size / original_size)
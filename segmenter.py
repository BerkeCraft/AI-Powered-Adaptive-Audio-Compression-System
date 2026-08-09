#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ses segmentasyonu ve etiketleme modülü
"""

import numpy as np

def segment_audio(audio_data, sample_rate, segment_duration_ms=20):
    """
    Ses verisini eşit uzunlukta segmentlere böler.
    
    Args:
        audio_data (numpy array): Normalize edilmiş ses verisi
        sample_rate (int): Örnekleme frekansı (Hz)
        segment_duration_ms (int): Segment süresi (milisaniye)
        
    Returns:
        list: Segmentlerin listesi (her segment numpy array)
    """
    # Segment uzunluğunu örnek sayısı cinsinden hesapla
    segment_length = int(sample_rate * segment_duration_ms / 1000.0)
    
    # Ses verisini segmentlere böl
    segments = []
    for i in range(0, len(audio_data), segment_length):
        segment = audio_data[i:i+segment_length]
        # Eğer son segment çok kısa ise, onu da ekle (istersek)
        if len(segment) > 0:
            segments.append(segment)
    
    return segments

def calculate_rms(segment):
    """
    Bir segment için RMS (Root Mean Square) değeri hesaplar.
    
    Args:
        segment (numpy array): Ses segmenti
        
    Returns:
        float: RMS değeri
    """
    if len(segment) == 0:
        return 0.0
    return np.sqrt(np.mean(segment.astype(np.float64) ** 2))

def determine_threshold(rms_values):
    """
    Otomatik eşik değeri belirler (ortalama RMS * 0.5).
    
    Args:
        rms_values (list): Segmentlerin RMS değerleri
        
    Returns:
        float: Eşik değeri
    """
    if len(rms_values) == 0:
        return 0.0
    return np.mean(rms_values) * 0.5

def label_segments(rms_values, threshold):
    """
    Segmentleri RMS değerlerine göre 'sessiz' veya 'aktif' olarak etiketler.
    
    Args:
        rms_values (list): Segmentlerin RMS değerleri
        threshold (float): Eşik değeri
        
    Returns:
        list: Etiketler (0: sessiz, 1: aktif)
    """
    labels = []
    for rms in rms_values:
        if rms < threshold:
            labels.append(0)  # sessiz
        else:
            labels.append(1)  # aktif
    return labels

def segment_and_label(audio_data, sample_rate, segment_duration_ms=20):
    """
    Ses verisini segmentlere böler ve etiketler.
    
    Args:
        audio_data (numpy array): Normalize edilmiş ses verisi
        sample_rate (int): Örnekleme frekansı (Hz)
        segment_duration_ms (int): Segment süresi (milisaniye)
        
    Returns:
        tuple: (labels, threshold)
            labels: Etiketler listesi (0: sessiz, 1: aktif)
            threshold: Otomatik eşik değeri
    """
    segments = segment_audio(audio_data, sample_rate, segment_duration_ms)
    rms_values = [calculate_rms(seg) for seg in segments]
    threshold = determine_threshold(rms_values)
    labels = label_segments(rms_values, threshold)
    return labels, threshold
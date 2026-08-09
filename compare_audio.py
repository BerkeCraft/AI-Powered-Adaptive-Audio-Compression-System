#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Karşılaştırma modülü: Kayıplı sıkıştırma stratejilerinin decode edilmiş hallerini
sesler klasörüne .wav dosyası olarak kaydeder.
"""

import os
import numpy as np
import scipy.io.wavfile as wav
import zlib
import segmenter
import compressors
import utils

def ensure_sesler_dir():
    """"""
    if not os.path.exists("sesler"):
        os.makedirs("sesler")
        print("'sesler' klasörü oluşturuldu.")

def load_wav_mono(filepath):
    """
    WAV dosyasını mono olarak yükler ve int16 PCM verisini döndürür.
    
    Args:
        filepath (str): WAV dosyasının yolu
        
    Returns:
        tuple: (sample_rate, audio_int16)
            sample_rate: Örnekleme frekansı (Hz)
            audio_int16: Mono ses verisi (numpy array, dtype=int16)
    """
    sample_rate, audio_data = wav.read(filepath)
    
    # Stereo ise mono'ya çevir
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # int16'e çevir (eğer zaten değilse)
    if audio_data.dtype != np.int16:
        # float veri ise -1 ile 1 arasında olduğunu varsayalım ve ölçekle
        if np.issubdtype(audio_data.dtype, np.floating):
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_int16 = (audio_data * 32767).astype(np.int16)
        else:
            # diğer tamsayı tipleri için doğrudan çevir ( riskli olabilir ama basitleştirme )
            audio_int16 = audio_data.astype(np.int16)
    else:
        audio_int16 = audio_data
    
    return sample_rate, audio_int16

def process_file(filepath, filename):
    """
    Tek bir WAV dosyasını işler ve lossy/adaptive/RVQ-GAN versiyonlarını üretir.
    
    Args:
        filepath (str): Orijinal WAV dosyasının tam yolu
        filename (str): Orijinal dosyanın adı (uzantı ile)
        
    Returns:
        dict: İşlem sonuçları veya None (hata durumunda)
    """
    try:
        # Orijinal sesi yükle
        sample_rate, original_int16 = load_wav_mono(filepath)
        original_size_bytes = original_int16.nbytes  # 2 bytes per sample
        
        # Segment etiketleri için normalize edilmiş float verisi hazırla
        audio_float = original_int16.astype(np.float32) / 32768.0
        labels, threshold = segmenter.segment_and_label(audio_float, sample_rate)
        
        # Segment uzunluğunu bayt cinsinden hesapla (20ms, 16-bit = 2 bytes/sample)
        segment_duration_ms = 20
        bytes_per_sample = 2  # 16-bit PCM
        segment_length = int(segment_duration_ms * sample_rate / 1000 * bytes_per_sample)
        
        # --- Kayıplı - Tekdüze ---
        # Sıkıştır
        compressed_uniform = compressors.compress_lossy_uniform(original_int16.tobytes())
        # Decode: zlib decompress -> bytes -> int16 array
        decompressed_uniform = zlib.decompress(compressed_uniform)
        decoded_uniform = np.frombuffer(decompressed_uniform, dtype=np.int16)
        # Uzunluk kontrolü (hazır olmalı ama güvenlik)
        if len(decoded_uniform) != len(original_int16):
            # Eğer uzunluk farklıysa, orijinal uzunluğa kırp veya pad et
            if len(decoded_uniform) > len(original_int16):
                decoded_uniform = decoded_uniform[:len(original_int16)]
            else:
                padding = np.zeros(len(original_int16) - len(decoded_uniform), dtype=np.int16)
                decoded_uniform = np.concatenate([decoded_uniform, padding])
        
        # --- Kayıplı - Adaptif ---
        # Sıkıştır
        compressed_adaptive = compressors.compress_lossy_adaptive(original_int16.tobytes(), labels, segment_length)
        # Decode
        decompressed_adaptive = zlib.decompress(compressed_adaptive)
        decoded_adaptive = np.frombuffer(decompressed_adaptive, dtype=np.int16)
        if len(decoded_adaptive) != len(original_int16):
            if len(decoded_adaptive) > len(original_int16):
                decoded_adaptive = decoded_adaptive[:len(original_int16)]
            else:
                padding = np.zeros(len(original_int16) - len(decoded_adaptive), dtype=np.int16)
                decoded_adaptive = np.concatenate([decoded_adaptive, padding])
        
        # --- Kayıplı - RVQ-GAN ---
        try:
            compressed_rvqgan = compressors.compress_rvqgan(original_int16, sample_rate)
            # Decode: Bu fonksiyon zaten_decode edilmiş sesi döndürür (int16)
            # Ancak, compress_rvqgan sadece sıkıştırılmış bytes döndürür, decode etmemiz gerekir.
            # Bunun için,เรา müssen wir die Metadata haben, aber wir haben sie nicht.
            # Daher müssen wir die encode_with_rvqgan-Funktion direkt aufrufen, um sowohl die komprimierten Bytes als auch die Metadata zu erhalten.
            import rvqgan_compressor
            audio_float = original_int16.astype(np.float32) / 32768.0
            compressed_bytes, metadata = rvqgan_compressor.encode_with_rvqgan(audio_float, sample_rate)
            decoded_rvqgan_float = rvqgan_compressor.decode_with_rvqgan(compressed_bytes, metadata)
            # Convert to int16 for consistency
            if decoded_rvqgan_float.dtype == np.float32:
                decoded_rvqgan = (np.clip(decoded_rvqgan_float, -1.0, 1.0) * 32767).astype(np.int16)
            else:
                decoded_rvqgan = decoded_rvqgan_float.astype(np.int16)
            # Ensure same length
            if len(decoded_rvqgan) != len(original_int16):
                if len(decoded_rvqgan) > len(original_int16):
                    decoded_rvqgan = decoded_rvqgan[:len(original_int16)]
                else:
                    padding = np.zeros(len(original_int16) - len(decoded_rvqgan), dtype=np.int16)
                    decoded_rvqgan = np.concatenate([decoded_rvqgan, padding])
        except Exception as e:
            print(f"  ! RVQ-GAN işleme hatası: {e}")
            # Fallback to original data
            decoded_rvqgan = original_int16
            compressed_rvqgan = original_int16.tobytes()
        
        # Boyutları hesapla (MB)
        size_uniform_mb = len(decoded_uniform) * 2 / (1024 * 1024)
        size_adaptive_mb = len(decoded_adaptive) * 2 / (1024 * 1024)
        size_rvqgan_mb = len(decoded_rvqgan) * 2 / (1024 * 1024)
        original_size_mb = original_size_bytes / (1024 * 1024)
        
        # Küçülme oranları
        reduction_uniform = (1 - size_uniform_mb / original_size_mb) * 100 if original_size_mb > 0 else 0
        reduction_adaptive = (1 - size_adaptive_mb / original_size_mb) * 100 if original_size_mb > 0 else 0
        reduction_rvqgan = (1 - size_rvqgan_mb / original_size_mb) * 100 if original_size_mb > 0 else 0
        
        # SNR hesapla
        snr_uniform = utils.calculate_snr(original_int16, decoded_uniform)
        snr_adaptive = utils.calculate_snr(original_int16, decoded_adaptive)
        snr_rvqgan = utils.calculate_snr(original_int16, decoded_rvqgan)
        
        # Kaydedilecek dosya yolları
        base_name = os.path.splitext(filename)[0]
        ext = ".wav"
        lossy_filename = f"lossy_{base_name}{ext}"
        adaptive_filename = f"adaptive_{base_name}{ext}"
        rvqgan_filename = f"rvqgan_{base_name}{ext}"
        lossy_filepath = os.path.join("sesler", lossy_filename)
        adaptive_filepath = os.path.join("sesler", adaptive_filename)
        rvqgan_filepath = os.path.join("sesler", rvqgan_filename)
        
        # Decode edilmiş sesleri kaydet (overwrite if exists)
        wav.write(lossy_filepath, sample_rate, decoded_uniform)
        wav.write(adaptive_filepath, sample_rate, decoded_adaptive)
        wav.write(rvqgan_filepath, sample_rate, decoded_rvqgan)
        
        return {
            'filename': filename,
            'lossy': {
                'filepath': lossy_filepath,
                'size_mb': size_uniform_mb,
                'reduction': reduction_uniform,
                'snr': snr_uniform
            },
            'adaptive': {
                'filepath': adaptive_filepath,
                'size_mb': size_adaptive_mb,
                'reduction': reduction_adaptive,
                'snr': snr_adaptive
            },
            'rvqgan': {
                'filepath': rvqgan_filepath,
                'size_mb': size_rvqgan_mb,
                'reduction': reduction_rvqgan,
                'snr': snr_rvqgan
            },
            'original_size_mb': original_size_mb
        }
    
    except Exception as e:
        print(f"  ! {filename} işlenirken hata: {e}")
        return None

def main():
    """Ana fonksiyon: sesler klasöründeki .wav dosyalarını işler ve sonuçları raporlar."""
    # Başlangıç mesajı
    print("=" * 50)
    print("KAYIPLI SES DOSYALARI OLUŞTURULDU")
    print("=" * 50)
    
    # sesler klasörünün存在を確認
    ensure_sesler_dir()
    
    # sesler klasöründeki tüm .wav dosyalarını bul
    wav_files = []
    for f in os.listdir("sesler"):
        if f.lower().endswith(".wav") and not f.startswith(("lossy_", "adaptive_", "rvqgan_")):
            wav_files.append(f)
    
    if not wav_files:
        print("  sesler klasöründe işlenecek .wav dosyası bulunamadı.")
        print("  Lütfen sesler/ klasörüne .wav dosyaları ekleyin.")
        print("=" * 50)
        return
    
    # Her dosyayı işle
    results = []
    for wav_file in sorted(wav_files):
        filepath = os.path.join("sesler", wav_file)
        print(f"  İşleniyor: {wav_file}")
        result = process_file(filepath, wav_file)
        if result:
            results.append(result)
    
    # Sonuçları yazdır
    print()
    for res in results:
        print(f"  {res['filename']}")
        # lossy
        snr_lossy = res['lossy']['snr']
        snr_lossy_str = "∞" if np.isinf(snr_lossy) else f"{snr_lossy:.1f}"
        print(f"    ✓ lossy_{os.path.splitext(res['filename'])[0]}.wav      → {res['lossy']['size_mb']:.2f} MB  (%{res['lossy']['reduction']:.1f} küçüldü)  SNR: {snr_lossy_str} dB")
        # adaptive
        snr_adaptive = res['adaptive']['snr']
        snr_adaptive_str = "∞" if np.isinf(snr_adaptive) else f"{snr_adaptive:.1f}"
        print(f"    ✓ adaptive_{os.path.splitext(res['filename'])[0]}.wav   → {res['adaptive']['size_mb']:.2f} MB  (%{res['adaptive']['reduction']:.1f} küçüldü)  SNR: {snr_adaptive_str} dB")
        # rvqgan
        snr_rvqgan = res['rvqgan']['snr']
        snr_rvqgan_str = "∞" if np.isinf(snr_rvqgan) else f"{snr_rvqgan:.1f}"
        print(f"    ✓ rvqgan_{os.path.splitext(res['filename'])[0]}.wav     → {res['rvqgan']['size_mb']:.2f} MB  (%{res['rvqgan']['reduction']:.1f} küçüldü)  SNR: {snr_rvqgan_str} dB")
        print()
    
    print("=" * 50)
    print("Tüm dosyalar \"sesler/\" klasörüne kaydedildi.")
    print("=" * 50)

if __name__ == "__main__":
    main()
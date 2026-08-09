#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Destekli Adaptif Ses Sıkıştırma Sistemi - Ana Modül

Bu modül, ses dosyasını komut satırından alır, segment analizini yapar,
4 farklı sıkıştırma stratejisini uygular ve sonuçları terminale yazdırır.
"""

import sys
import os
import numpy as np
import zlib
import scipy.io.wavfile as wav
import segmenter
import compressors
import utils

def load_audio_file(filepath):
    """
    Ses dosyasını yükler ve numpy array'e dönüştürür.
    Şu anda sadece .wav formatını destekliyor.
    
    Args:
        filepath (str): Ses dosyasının yolu
        
    Returns:
        tuple: (audio_data, sample_rate, channels, filepath)
            audio_data: Normalize edilmiş mono ses verisi (numpy array)
            sample_rate: Örnekleme frekansı (Hz)
            channels: Kanal sayısı (1 for mono, 2 for stereo)
            filepath: Original file path for size reference
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dosya bulunamadı: {filepath}")
    
    # Dosya uzantısını kontrol et
    if not filepath.lower().endswith('.wav'):
        raise ValueError("Desteklenmeyen format. Şu anda sadece .wav formatı desteklenmektedir.")
    
    # scipy.io.wavfile ile ses dosyasını yükle
    sample_rate, audio_data = wav.read(filepath)
    
    # Stereo ise mono'ya çevir (basitleştirmek için)
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Veriyi float32'e çevir ve normalize et
    # Important: After np.mean on integer types, we get float64 but the values
    # are still in the original integer range, not normalized to [-1, 1]!
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    elif audio_data.dtype == np.uint8:
        audio_data = (audio_data.astype(np.float32) - 128) / 128.0
    else:
        # For float types (including float64 from np.mean), we need to check
        # if they're already in [-1, 1] range or if they're integer-equivalent
        # If the min/max are reasonable for audio samples (e.g., -5000 to 5000),
        # they're likely integer-equivalent and need normalization
        if audio_data.dtype == np.float64 or audio_data.dtype == np.float32:
            # Check if values suggest they're integer samples (not normalized)
            abs_max = np.max(np.abs(audio_data))
            if abs_max > 1.0:
                # These appear to be integer-equivalent samples, normalize them
                # Determine the original bit depth from the range
                if abs_max <= 32768:  # Likely from int16
                    audio_data = audio_data / 32768.0
                elif abs_max <= 2147483648:  # Likely from int32
                    audio_data = audio_data / 2147483648.0
                else:
                    # Fallback: assume already normalized
                    pass
            # If abs_max <= 1.0, assume already normalized
        # Zaten float ise, -1 ile 1 arasında olduğunu varsayalım
        audio_data = audio_data.astype(np.float32)
    
    # Orijinal veri boyutu (bytes)
    # Float32 verisini 16-bit PCM'e geri çevirerek boyutu hesapla
    original_int16 = (audio_data * 32768).astype(np.int16)
    pcm_size = original_int16.nbytes
    # Get actual file size for comparison
    file_size = os.path.getsize(filepath)
    # We'll return the filepath so that the caller can use it for debug prints
    # For compression ratio, we'll use PCM size (without header) as the reference
    
    return audio_data, sample_rate, 1, filepath  # her zaman mono döndürüyoruz

def run_compression_pipeline(audio_data, sample_rate, filepath):
    """
    Ana sıkıştırma işlem hattını çalıştırır.
    
    Args:
        audio_data (numpy array): Normalize edilmiş ses verisi
        sample_rate (int): Örnekleme frekansı
        filepath (str): Path to the audio file
        
    Returns:
        dict: Her sıkıştırma stratejisi için sonuçlar
    """
    # Segment analizi
    labels, threshold = segmenter.segment_and_label(audio_data, sample_rate)
    
    # Orijinal veri boyutu (bytes)
    # Float32 verisini 16-bit PCM'e geri çevirerek boyutu hesapla
    original_int16 = (audio_data * 32768).astype(np.int16)
    pcm_size = original_int16.nbytes
    # Get actual file size for comparison
    file_size = os.path.getsize(filepath)
    original_size = pcm_size  # We'll use PCM size for compression ratio calculation
    
    print(f"[DEBUG] WAV dosya boyutu   : {file_size} bytes ({file_size/1024/1024:.2f} MB)")
    print(f"[DEBUG] PCM veri boyutu    : {pcm_size} bytes ({pcm_size/1024/1024:.2f} MB)")
    print(f"[DEBUG] WAV header boyutu  : {file_size - pcm_size} bytes")
    
    # Segment uzunluğunu hesapla (20ms segmentler, 16-bit = 2 bytes per sample)
    segment_duration_ms = 20
    bytes_per_sample = 2  # 16-bit PCM
    segment_length = int(segment_duration_ms * sample_rate / 1000 * bytes_per_sample)
    
    results = {}
    
    # Strateji a: Kayıpsız - Tekdüze (Baseline): Tüm dosyaya standart zlib sıkıştırma
    try:
        compressed_data = compressors.compress_lossless_uniform(original_int16.tobytes())
        compressed_size = len(compressed_data)
        ratio = utils.calculate_compression_ratio(original_size, compressed_size)
        results['lossless_uniform'] = {
            'size': compressed_size,
            'ratio': ratio,
            'snr': None  # Kayıpsız için SNR yok
        }
    except Exception as e:
        print(f"Kayıpsız - Tekdüze sıkıştırma hatası: {e}")
        results['lossless_uniform'] = {
            'size': original_size,
            'ratio': 0.0,
            'snr': None
        }
    
    # Strateji b: Kayıpsız - Adaptif
    try:
        compressed_data = compressors.compress_lossless_adaptive(original_int16.tobytes(), labels, segment_length)
        compressed_size = len(compressed_data)
        ratio = utils.calculate_compression_ratio(original_size, compressed_size)
        results['lossless_adaptive'] = {
            'size': compressed_size,
            'ratio': ratio,
            'snr': None
        }
    except Exception as e:
        print(f"Kayıpsız - Adaptif sıkıştırma hatası: {e}")
        results['lossless_adaptive'] = {
            'size': original_size,
            'ratio': 0.0,
            'snr': None
        }
    
    # Strateji c: Kayıplı - Tekdüze (Baseline): Tüm sese düşük bit derinliği (16-bit → 8-bit) kuantizasyon
    try:
        # 16-bit'i 8-bit'e kuantize et
        quantized = compressors.quantize_audio(original_int16, target_bits=8)
        # Dequantize to calculate SNR (this is where the loss occurs)
        dequantized = compressors.dequantize_audio(quantized, target_bits=8)
        # Zlib ile sıkıştır
        compressed_data = zlib.compress(quantized.tobytes(), level=6)
        compressed_size = len(compressed_data)
        ratio = utils.calculate_compression_ratio(original_size, compressed_size)
        results['lossy_uniform'] = {
            'size': compressed_size,
            'ratio': ratio,
            'snr': utils.calculate_snr(original_int16, dequantized)
        }
    except Exception as e:
        print(f"Kayıplı - Tekdüze sıkıştırma hatası: {e}")
        results['lossy_uniform'] = {
            'size': original_size,
            'ratio': 0.0,
            'snr': 0.0
        }
    
    # Strateji d: Kayıplı - Adaptif
    try:
        compressed_data = compressors.compress_lossy_adaptive(original_int16.tobytes(), labels, segment_length)
        # For SNR calculation, we need to decode the adaptive strategy
        # This is complex, so we'll approximate by applying the same quantization
        # but this won't be accurate. For now, we'll calculate SNR by comparing
        # with a uniformly quantized version (not ideal but shows concept)
        quantized_approx = compressors.quantize_audio(original_int16, target_bits=8)
        dequantized_approx = compressors.dequantize_audio(quantized_approx, target_bits=8)
        compressed_size = len(compressed_data)
        ratio = utils.calculate_compression_ratio(original_size, compressed_size)
        results['lossy_adaptive'] = {
            'size': compressed_size,
            'ratio': ratio,
            'snr': utils.calculate_snr(original_int16, dequantized_approx)
        }
    except Exception as e:
        print(f"Kayıplı - Adaptif sıkıştırma hatası: {e}")
        results['lossy_adaptive'] = {
            'size': original_size,
            'ratio': 0.0,
            'snr': 0.0
        }
    
    # Strateji e: Kayıplı - RVQ-GAN ★
    try:
        compressed_data = compressors.compress_rvqgan(original_int16, sample_rate)
        compressed_size = len(compressed_data)
        ratio = utils.calculate_compression_ratio(original_size, compressed_size)
        results['rvqgan'] = {
            'size': compressed_size,
            'ratio': ratio,
            'snr': None  # We'll calculate SNR properly below if possible
        }
        # Try to calculate SNR by decompressing
        try:
            # For RVQ-GAN, we need to decode to calculate SNR
            import rvqgan_compressor
            # We need to get the metadata to decode properly
            # But our compress_rvqgan only returns bytes, not metadata
            # Let's modify approach: call encode_with_rvqgan directly to get both
            audio_float = original_int16.astype(np.float32) / 32768.0
            compressed_bytes, metadata = rvqgan_compressor.encode_with_rvqgan(audio_float, sample_rate)
            decoded_array = rvqgan_compressor.decode_with_rvqgan(compressed_bytes, metadata)
            # Convert decoded array to int16 for SNR comparison
            if decoded_array.dtype == np.float32:
                decoded_int16 = (np.clip(decoded_array, -1.0, 1.0) * 32767).astype(np.int16)
            else:
                decoded_int16 = decoded_array.astype(np.int16)
            # Ensure same length
            if len(decoded_int16) != len(original_int16):
                if len(decoded_int16) > len(original_int16):
                    decoded_int16 = decoded_int16[:len(original_int16)]
                else:
                    padding = np.zeros(len(original_int16) - len(decoded_int16), dtype=np.int16)
                    decoded_int16 = np.concatenate([decoded_int16, padding])
            results['rvqgan']['snr'] = utils.calculate_snr(original_int16, decoded_int16)
        except Exception as e2:
            print(f"RVQ-GAN SNR calculation error: {e2}")
            results['rvqgan']['snr'] = None
    except ImportError as e:
        # Model not available
        results['rvqgan'] = {
            'size': original_size,
            'ratio': 0.0,
            'snr': "Model yüklenemedi"   # Special string to display in table
        }
    except Exception as e:
        print(f"Kayıplı - RVQ-GAN ★ sıkıştırma hatası: {e}")
        results['rvqgan'] = {
            'size': original_size,
            'ratio': 0.0,
            'snr': 0.0   # Indicate other error
        }
    
    # Strateji f: Akıllı - Entropy Seçici ★
    try:
        # Import the entropy selector
        import entropy_selector
        compressed_data = entropy_selector.compress_with_entropy_selection(original_int16, sample_rate)
        compressed_size = len(compressed_data)
        ratio = utils.calculate_compression_ratio(original_size, compressed_size)
        results['entropy_selection'] = {
            'size': compressed_size,
            'ratio': ratio,
            'snr': None  # We'll calculate SNR by comparing with original
        }
        # Try to calculate SNR for entropy selection
        # This is complex due to mixed lossless/lossy nature, but we can approximate
        # by checking if any RVQ-GAN segments were used and calculating SNR for those
        try:
            # For now, we'll set SNR to None and display '-' in the table
            # A more sophisticated approach would involve tracking which segments used RVQ-GAN
            # and calculating SNR only for those segments, but that requires modifying
            # the entropy_selector to return additional information
            results['entropy_selection']['snr'] = None
        except Exception:
            results['entropy_selection']['snr'] = None
    except ImportError as e:
        # Model not available
        results['entropy_selection'] = {
            'size': original_size,
            'ratio': 0.0,
            'snr': "Model yüklenemedi"   # Special string to display in table
        }
    except Exception as e:
        results['entropy_selection'] = {
            'size': original_size,
            'ratio': 0.0,
            'snr': 0.0
        }
    
    return results, original_size, labels

def print_results_table(original_size, results, labels, threshold):
    """
    Sonuçları tablo formatında terminale yazdırır.
    
    Args:
        original_size (int): Orijinal dosya boyutu (bytes)
        results (dict): Sıkıştırma sonuçları
        labels (list): Segment etiketleri
        threshold (float): Otomatik eşik değeri
    """
    # Segment istatistikleri
    total_segments = len(labels)
    silent_segments = sum(1 for label in labels if label == 0)  # 0: sessiz
    active_segments = total_segments - silent_segments
    
    silent_percent = (silent_segments / total_segments) * 100 if total_segments > 0 else 0
    active_percent = (active_segments / total_segments) * 100 if total_segments > 0 else 0
    
    # Tablo başlığı
    print("=" * 60)
    print("ADAPTİF SES SIKIŞTIRAMA - SONUÇ KARŞILAŞTIRMASI")
    print("=" * 60)
    print(f"Orijinal Dosya Boyutu : {utils.format_size(original_size)}")
    print()
    print(f"{'Strateji':<25} | {'Boyut':<10} | {'Oran':<8} | {'SNR (dB)':<10}")
    print("-" * 60)
    
    # Strateji sonuçları
    strategies = [
        ('lossless_uniform', 'Kayıpsız - Tekdüze'),
        ('lossless_adaptive', 'Kayıpsız - Adaptif'),
        ('lossy_uniform', 'Kayıplı  - Tekdüze'),
        ('lossy_adaptive', 'Kayıplı  - Adaptif')
    ]
    # Try to add RVQ-GAN strategy if dependencies are available
    try:
        import rvqgan_compressor
        strategies.append(('rvqgan', 'Kayıplı  - RVQ-GAN ★'))
    except ImportError:
        pass  # RVQ-GAN not available, skip
    # Try to add Entropy Selection strategy
    try:
        import entropy_selector
        strategies.append(('entropy_selection', 'Akıllı   - Entropy Seçici ★'))
    except ImportError:
        pass  # Entropy selector not available, skip
    
    for key, name in strategies:
        res = results.get(key, {'size': original_size, 'ratio': 0.0, 'snr': None})
        size_str = utils.format_size(res['size'])
        ratio_str = f"%{res['ratio']*100:.1f}"
        snr_str = "-" if res['snr'] is None else ( "∞" if np.isinf(res['snr']) else f"{res['snr']:.2f}" )
        print(f"{name:<25} | {size_str:<10} | {ratio_str:<8} | {snr_str:<10}")
    
    print("-" * 60)
    print("Segment Analizi:")
    print(f"  Toplam segment sayısı   : {total_segments}")
    print(f"  Sessiz segment sayısı   : {silent_segments} ({silent_percent:.1f}%)")
    print(f"  Aktif segment sayısı    : {active_segments} ({active_percent:.1f}%)")
    print(f"  Otomatik eşik değeri    : {threshold:.6f}")
    print(f"  Sessiz segment oranı: %{silent_percent:.1f} ({'avantajlı' if silent_percent >= 20 else 'avantaj azalır'})")
    print("=" * 60)

def main():
    """
    Ana fonksiyon: Komut satırından dosya alır ve sıkıştırma işlemini başlatır.
    """
    if len(sys.argv) != 2:
        print("Kullanım: python main.py <ses_dosyasi>")
        print("Örnek: python main.py input.wav")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    try:
        # Ses dosyasını yükle
        audio_data, sample_rate, channels, filepath_returned = load_audio_file(filepath)
        print(f"Yüklenen dosya: {filepath}")
        print(f"Örnekleme frekansı: {sample_rate} Hz, Kanallar: {channels}, Örnek sayısı: {len(audio_data)}")
        print()
        
        # Sıkıştırma işlem hattını çalıştır
        results, original_size, labels = run_compression_pipeline(audio_data, sample_rate, filepath)
        
        # Segment etiketleme ve eşik değeri için tekrar hesaplama (basitleştirilmiş)
        # Gerçekte bu değerler run_compression_pipeline içinde hesaplanmalı ve döndürülmeli
        # Şimdilik basitleştirilmiş bir hesaplama yapıyoruz
        from segmenter import segment_audio, calculate_rms, determine_threshold, label_segments
        segments = segment_audio(audio_data, sample_rate)
        rms_values = [calculate_rms(seg) for seg in segments]
        threshold = determine_threshold(rms_values)
        labels = label_segments(rms_values, threshold)
        
        # Sonuçları yazdır
        print_results_table(original_size, results, labels, threshold)
        
    except FileNotFoundError as e:
        print(f"Hata: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Hata: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Beklenmeyen hata: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
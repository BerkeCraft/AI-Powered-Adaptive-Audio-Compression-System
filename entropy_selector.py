#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entropy-based compression strategy selection modülü
"""

import numpy as np
import math
from collections import Counter
import segmenter

def calculate_entropy(segment_bytes):
    """
    Bir segment'in Shannon entropy değerini hesaplar.
    
    Args:
        segment_bytes (bytes): Segment verisi
        
    Returns:
        float: Shannon entropy değeri (0-8 bit arası)
    """
    if not segment_bytes:
        return 0.0
    
    # Byte frekanslarını hesapla
    freq = {}
    for byte_val in segment_bytes:
        freq[byte_val] = freq.get(byte_val, 0) + 1
    
    # Olasılıkları hesapla ve entropy'yi topla
    entropy = 0.0
    total = len(segment_bytes)
    for count in freq.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    return entropy

def select_algorithm_for_segment(segment_bytes, is_silent=False):
    """
    Bir segment için entropy değerine göre uygun sıkıştırma algoritmasını seçer.
    
    Args:
        segment_bytes (bytes): Segment verisi
        is_silent (bool): Segment sessiz mi? (RMS < eşik)
        
    Returns:
        tuple: (algorithm_name, explanation)
            algorithm_name: Seçilen algoritma adı ("RLE", "Huffman", "Arithmetic", "RVQ-GAN")
            explanation: Algoritma seçiminin açıklaması
    """
    # Sessiz segmentler her zaman RLE
    if is_silent:
        return ("RLE", "Sessiz bölge tespit edildi. Enerji eşiğinin altında, RLE sıfır değerlerini çok verimli sıkıştırır.")
    
    # Aktif segmentler için entropy hesapla
    entropy = calculate_entropy(segment_bytes)
    
    if entropy < 1.0:
        return ("RLE", f"Segment çok düzenli ve tekrarlı. Entropy: {entropy:.2f}. RLE bu yapıyı en verimli şekilde sıkıştırır.")
    elif entropy < 3.5:
        return ("Huffman", f"Segment orta düzey karmaşıklıkta. Entropy: {entropy:.2f}. Bazı değerler diğerlerinden çok daha sık geçiyor, Huffman bu frekans farkını avantaja çevirir.")
    elif entropy < 6.0:
        return ("Arithmetic", f"Segment yüksek karmaşıklıkta. Entropy: {entropy:.2f}. Huffman'ın tam sayı bit sınırı yetersiz kalır, entropy sınırına daha yakın sıkıştırma gerekir (zlib level=9 ile temsil edilir).")
    else:
        return ("RVQ-GAN", f"Segment çok yüksek entropylu ve rastlantısal görünüyor. Entropy: {entropy:.2f}. Klasik algoritmalar yetersiz kalır. Sinir ağı tabanlı RVQ-GAN bu tür karmaşık yapıları öğrenerek sıkıştırır.")

def analyze_and_select(segments, labels):
    """
    Tüm segmentleri analiz eder ve her biri için uygun algoritmayı seçer.
    
    Args:
        segments (list of numpy array): Segmentlerin listesi
        labels (list of int): Segment etiketleri (0: sessiz, 1: aktif)
        
    Returns:
        list of dict: Her segment için analiz sonucu
            Her dict şu anahtarları içerir:
                - segment_id: int
                - label: "sessiz" veya "aktif"
                - entropy: float (2 ondalık yuvarlanmış)
                - algorithm: "RLE", "Huffman", "Arithmetic", "RVQ-GAN"
                - explanation: str (açıklama metni)
    """
    results = []
    for i, (segment, label) in enumerate(zip(segments, labels)):
        # Segmenti bytes'a çevir (16-bit PCM varsayımı)
        # segment numpy array, float32, -1 ile 1 arasında
        if segment.dtype == np.float32:
            # float32 to int16
            segment_int16 = (np.clip(segment, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            # Zaten int16 veya başka bir tamsayı tipindeyse, doğrudan çevir
            segment_int16 = segment.astype(np.int16)
        segment_bytes = segment_int16.tobytes()
        
        is_silent = (label == 0)
        algorithm, explanation = select_algorithm_for_segment(segment_bytes, is_silent)
        entropy = calculate_entropy(segment_bytes)
        
        results.append({
            "segment_id": i,
            "label": "sessiz" if is_silent else "aktif",
            "entropy": round(entropy, 2),
            "algorithm": algorithm,
            "explanation": explanation
        })
    
    return results

def compress_with_entropy_selection(audio_data, sample_rate):
    """
    Entropy tabanlı adaptif sıkıştırma stratejisini uygular.
    
    Args:
        audio_data (numpy array): Normalize edilmiş mono ses verisi (float32)
        sample_rate (int): Örnekleme frekansı
        
    Returns:
        bytes: Sıkıştırılmış veri
    """
    # Import gereken modüller
    import segmenter
    import compressors
    import zlib
    import rvqgan_compressor
    
    # Segmentasyon ve etiketleme
    segments = segmenter.segment_audio(audio_data, sample_rate)
    # Normalize edilmiş float veriyi segmenter ile zaten float olarak alıyoruz, labels için de aynı kullanıyoruz
    labels, _ = segmenter.segment_and_label(audio_data, sample_rate)
    
    # Her segment için analiz ve algoritma seçimi
    analysis = analyze_and_select(segments, labels)
    
    # Her segmenti seçilen algoritmaya göre sıkıştır
    compressed_segments = bytearray()
    
    # For SNR calculation of RVQ-GAN segments
    decoded_segments = []  # Store decoded segments for RVQ-GAN
    
    for i, (segment, result) in enumerate(zip(segments, analysis)):
        # Segmenti bytes'a çevir
        if segment.dtype == np.float32:
            segment_int16 = (np.clip(segment, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            segment_int16 = segment.astype(np.int16)
        segment_bytes = segment_int16.tobytes()
        
        algorithm = result["algorithm"]
        label = result["label"]  # "sessiz" veya "aktif"
        
        # Seçilen algoritmaya göre sıkıştır
        if label == "sessiz":
            # For silent segments, we use a more aggressive compression for lossless algorithms? 
            # But note: the entropy selector doesn't differentiate between silent and active for the algorithm choice beyond the silent->RLE rule.
            # We already forced RLE for silent segments.
            # For silent, we can use:
            #   RLE: our RLE then zlib level=9
            #   Huffman: our Huffman then zlib level=9
            #   Arithmetic: zlib level=9
            #   RVQ-GAN: rvqgan_compressor (which is lossy, and we don't have a silent/active distinction in RVQ-GAN call)
            # But note: the RVQ-GAN compressor in our rvqgan_compressor.py does not take labels, it just compresses the entire array.
            # However, we are calling it per segment. We'll use the same function.
            if algorithm == "RLE":
                compressed = compressors.rle_encode(segment_bytes)
                compressed = zlib.compress(compressed, level=9)
            elif algorithm == "Huffman":
                try:
                    compressed = compressors.huffman_encode(segment_bytes)
                    compressed = zlib.compress(compressed, level=9)
                except NotImplementedError:
                    compressed = zlib.compress(segment_bytes, level=9)
            elif algorithm == "Arithmetic":
                compressed = zlib.compress(segment_bytes, level=9)
            else:  # RVQ-GAN
                try:
                    compressed = rvqgan_compressor.compress_rvqgan(segment_int16, sample_rate)
                except Exception as e:
                    # If RVQ-GAN compression fails, fall back to zlib
                    print(f"RVQ-GAN compression failed in entropy selector: {e}")
                    compressed = zlib.compress(segment_bytes, level=9)
        else:  # active
            if algorithm == "RLE":
                compressed = compressors.rle_encode(segment_bytes)
                compressed = zlib.compress(compressed, level=6)
            elif algorithm == "Huffman":
                try:
                    compressed = compressors.huffman_encode(segment_bytes)
                    compressed = zlib.compress(compressed, level=6)
                except NotImplementedError:
                    compressed = zlib.compress(segment_bytes, level=6)
            elif algorithm == "Arithmetic":
                compressed = zlib.compress(segment_bytes, level=6)
            else:  # RVQ-GAN
                try:
                    compressed = rvqgan_compressor.compress_rvqgan(segment_int16, sample_rate)
                except Exception as e:
                    # If RVQ-GAN compression fails, fall back to zlib
                    print(f"RVQ-GAN compression failed in entropy selector: {e}")
                    compressed = zlib.compress(segment_bytes, level=6)
            
            compressed_segments.extend(compressed)
    
    # TODO: We would need to return both compressed data and SNR info
    # For now, we'll just return the compressed data as before
    # The SNR calculation will be done in main.py similar to RVQ-GAN strategy
    return bytes(compressed_segments)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Kullanım: python entropy_selector.py <ses_dosyasi>")
        sys.exit(1)
    filepath = sys.argv[1]
    # Load the audio file
    import scipy.io.wavfile as wav
    import numpy as np
    sample_rate, audio_data = wav.read(filepath)
    # Convert to mono float32 as in main.py
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
    # Now run the entropy selection and print the analysis
    from segmenter import segment_audio, segment_and_label
    segments = segment_audio(audio_data, sample_rate)
    labels, _ = segment_and_label(audio_data, sample_rate)
    analysis = analyze_and_select(segments, labels)
    print("Entropy Analizi - Strateji Dağılımı")
    print("="*50)
    # We'll print a summary similar to what is requested in the terminal output for main.py
    total_segments = len(analysis)
    # Count by algorithm
    algo_counts = Counter([r['algorithm'] for r in analysis])
    # Also compute average entropy per algorithm
    algo_entropy = {}
    for algo in ["RLE", "Huffman", "Arithmetic", "RVQ-GAN"]:
        entropies = [r['entropy'] for r in analysis if r['algorithm'] == algo]
        if entropies:
            algo_entropy[algo] = sum(entropies)/len(entropies)
        else:
            algo_entropy[algo] = 0.0
    print(f"Toplam segment     : {total_segments}")
    print(f"RLE seçildi        : {algo_counts.get('RLE',0)} segment (%{algo_counts.get('RLE',0)/total_segments*100:.1f}) — ort. entropy: {algo_entropy['RLE']:.2f}")
    print(f"Huffman seçildi    : {algo_counts.get('Huffman',0)} segment (%{algo_counts.get('Huffman',0)/total_segments*100:.1f}) — ort. entropy: {algo_entropy['Huffman']:.2f}")
    print(f"Arithmetic seçildi : {algo_counts.get('Arithmetic',0)} segment (%{algo_counts.get('Arithmetic',0)/total_segments*100:.1f}) — ort. entropy: {algo_entropy['Arithmetic']:.2f}")
    print(f"RVQ-GAN seçildi    : {algo_counts.get('RVQ-GAN',0)} segment (%{algo_counts.get('RVQ-GAN',0)/total_segments*100:.1f}) — ort. entropy: {algo_entropy['RVQ-GAN']:.2f}")
    print("="*50)
    # Find the most complex and simplest segments
    if analysis:
        # Most complex: highest entropy
        max_entropy_segment = max(analysis, key=lambda x: x['entropy'])
        min_entropy_segment = min(analysis, key=lambda x: x['entropy'])
        print(f"En karmaşık segment: #{max_entropy_segment['segment_id']}  entropy: {max_entropy_segment['entropy']:.2f} → {max_entropy_segment['algorithm']}")
        print(f"En düzenli segment : #{min_entropy_segment['segment_id']}  entropy: {min_entropy_segment['entropy']:.2f} → {min_entropy_segment['algorithm']}")
    print("="*50)
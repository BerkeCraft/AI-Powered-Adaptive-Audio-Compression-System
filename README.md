# 🎵 AI-Powered Adaptive Audio Compression System

> Classic algorithms + Artificial Intelligence = 98.2% compression ratio

An adaptive audio compression system built with Python, using Shannon entropy 
analysis to select the optimal algorithm per segment. Includes Meta's EnCodec 
(RVQ-GAN) neural codec and a Tkinter GUI.

---

## 🚀 Features

- `.wav` and `.mp3` format support
- Automatic 20ms segmentation with RMS energy analysis
- Shannon entropy-based intelligent algorithm selection
- 6 different compression strategies
- Tkinter graphical user interface
- Lossy audio export and comparison tool

---

## 📊 Compression Strategies

| Strategy | Method | Ratio |
|---|---|---|
| Lossless - Uniform | zlib | 21.2% |
| Lossless - Adaptive | RLE + Huffman + zlib | 17.6% |
| Lossy - Uniform | 8-bit quantization + zlib | 91.3% |
| Lossy - Adaptive | 4/12-bit quantization + zlib | 66.0% |
| Lossy - RVQ-GAN ★ | EnCodec + Huffman + zlib | 98.1% |
| Smart - Entropy Selector ★ | Automatic selection | 98.2% |

---

## 🧠 Entropy Selector Logic

| Entropy (H) | Selected Algorithm | Reason |
|---|---|---|
| H < 1.0 | RLE | Highly repetitive, ordered data |
| 1.0 ≤ H < 3.5 | Huffman | Distinct frequency differences |
| 3.5 ≤ H < 6.0 | Arithmetic (zlib) | High complexity |
| H ≥ 6.0 | RVQ-GAN | Chaotic data, neural net required |
| RMS < threshold | RLE | Silent segment detected |

---

## 📁 Project Structure
├── main.py # Main pipeline and comparison table
├── segmenter.py # RMS analysis and segment labeling
├── compressors.py # RLE, Huffman, quantization
├── utils.py # SNR calculation, helper functions
├── gui.py # Tkinter GUI
├── rvqgan_compressor.py # EnCodec RVQ-GAN integration
├── entropy_selector.py # Shannon entropy decision system
├── compare_audio.py # Lossy audio export tool
└── audio/ # Test audio files
---

## ⚙️ Installation

```bash
git clone https://github.com/your_username/adaptive-audio-compression
cd adaptive-audio-compression
pip install numpy scipy pydub encodec torch torchaudio
```

---

## 🖥️ Usage

**With GUI:**
```bash
python gui.py
```

**With terminal:**
```bash
python main.py audio_file.wav
```

**Export lossy audio files:**
```bash
python compare_audio.py
```

---

## 📈 Test Results

Test file: `GuitarAmbition.wav` (9.89 MB)

| Metric | Value |
|---|---|
| Best compression ratio | **98.2%** (Entropy Selector) |
| SNR (lossy strategies) | **23.03 dB** (human ear threshold: 20 dB) |
| RVQ-GAN output size | **195 KB** (from 9.89 MB) |

---
## 🔬 Technical Details

- **RLE:** Compress only if consecutive repetitions ≥ 3, otherwise skip
- **Huffman:** Frequency table → binary tree → encode → zlib
- **Quantization:** 16-bit → 4/8/12-bit, irreversible
- **RVQ-GAN:** EnCodec vectors → tobytes() → zlib
- **Entropy:** H = -Σ p(x) · log₂(p(x))

---

## 🔗 Inspiration

This project draws inspiration from the MP3 format's core engineering principles:

| MP3 Component | This Project | Status |
|---|---|---|
| Huffman coding | compressors.py | ✅ Implemented |
| Quantization | compressors.py | ✅ Implemented |
| Silent/active separation | segmenter.py | ✅ Implemented |
| MDCT frequency transform | - | Out of scope |
| Psychoacoustic model | - | Out of scope |

---

## 📚 References

- [Meta EnCodec](https://github.com/facebookresearch/encodec)
- Huffman, D. A. (1952). *A Method for the Construction of Minimum-Redundancy Codes.*
- Shannon, C. E. (1948). *A Mathematical Theory of Communication.*
- Défossez, A. et al. (2022). *High Fidelity Neural Audio Compression.* Meta AI Research.

---

## 📝 License

MIT License

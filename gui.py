#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tkinter GUI for Adaptive Audio Compression System
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import main as main_module
import os
import numpy as np


class AdaptiveAudioCompressionGUI:
    def __init__(self, root):
        print("GUI Initializing...")
        self.root = root
        self.root.title("Adaptif Ses Sıkıştırma Sistemi")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        # Variables
        self.selected_file = tk.StringVar()  # full path
        self.display_file = tk.StringVar()   # displayed basename
        self.is_processing = False
        
        # Create GUI elements
        self.create_widgets()
        print("GUI Initialized.")
        
    def create_widgets(self):
        # Title Label
        title_label = tk.Label(
            self.root, 
            text="Adaptif Ses Sıkıştırma Sistemi", 
            font=("Arial", 16, "bold"),
            pady=10
        )
        title_label.pack()
        
        # File Selection Frame
        file_frame = tk.Frame(self.root, pady=10)
        file_frame.pack(fill="x", padx=20)
        
        select_button = tk.Button(
            file_frame,
            text="Dosya Seç",
            command=self.select_file,
            width=15
        )
        select_button.pack(side="left")
        
        file_label = tk.Label(
            file_frame,
            textvariable=self.display_file,
            wraplength=400,
            justify="left"
        )
        file_label.pack(side="left", padx=10)
        
        # Compress Button
        self.compress_button = tk.Button(
            self.root,
            text="Sıkıştır",
            command=self.start_compression,
            width=20,
            height=2,
            bg="#003366",
            fg="white",
            font=("Arial", 10, "bold")
        )
        self.compress_button.pack(pady=20)
        
        # Results Table Frame
        results_frame = tk.Frame(self.root)
        results_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Create Treeview for results
        self.tree = ttk.Treeview(
            results_frame,
            columns=("Strateji", "Boyut", "Oran", "SNR (dB)"),
            show="headings",
            height=8
        )
        
        # Define headings
        self.tree.heading("Strateji", text="Strateji")
        self.tree.heading("Boyut", text="Boyut")
        self.tree.heading("Oran", text="Oran")
        self.tree.heading("SNR (dB)", text="SNR (dB)")
        
        # Define column widths
        self.tree.column("Strateji", width=150)
        self.tree.column("Boyut", width=100)
        self.tree.column("Oran", width=100)
        self.tree.column("SNR (dB)", width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack treeview and scrollbar
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Summary Info Label
        self.summary_label = tk.Label(
            self.root,
            text="Orijinal: 0.00 MB | En iyi oran: %0.0 (Henüz işlem yapılmadı)",
            font=("Arial", 9),
            pady=5
        )
        self.summary_label.pack()
        
        # Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set("Hazır")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def select_file(self):
        """Open file dialog to select audio file"""
        filetypes = [
            ("WAV files", "*.wav"),
            ("MP3 files", "*.mp3"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Ses Dosyası Seçin",
            filetypes=filetypes
        )
        
        if filename:
            self.selected_file.set(filename)  # store full path
            self.display_file.set(os.path.basename(filename))  # display basename
            self.status_var.set(f"Dosya yüklendi: {os.path.basename(filename)}")
        else:
            self.status_var.set("Dosya seçilmedi")
    
    def start_compression(self):
        """Start compression process in a separate thread"""
        if not self.selected_file.get():
            messagebox.showwarning("Uyarı", "Lütfen önce bir dosya seçin!")
            return
        
        if self.is_processing:
            messagebox.showinfo("Bilgi", "Zaten bir işlem devam ediyor!")
            return
        
        # Disable button and change text
        self.compress_button.config(
            text="İşleniyor...",
            state="disabled",
            bg="#cccccc"
        )
        
        # Update status
        self.status_var.set("Sıkıştırılıyor...")
        
        # Clear previous results
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Start compression in separate thread
        self.is_processing = True
        thread = threading.Thread(target=self.run_compression)
        thread.daemon = True
        thread.start()
    
    def run_compression(self):
        """Run the compression process (called in separate thread)"""
        try:
            # Get the already selected file path
            filepath = self.selected_file.get()
            if not filepath:
                # Resolve relative path to absolute
                filepath = os.path.abspath(filepath)
                self.root.after(0, self.compression_cancelled)
                return
            
            # Run the main compression pipeline
            # Import and use main functions directly
            audio_data, sample_rate, channels, _ = main_module.load_audio_file(filepath)
            results, original_size, labels = main_module.run_compression_pipeline(audio_data, sample_rate, filepath)
            
            # Get segment info for display
            from segmenter import segment_audio, calculate_rms, determine_threshold, label_segments
            segments = segment_audio(audio_data, sample_rate)
            rms_values = [calculate_rms(seg) for seg in segments]
            threshold = determine_threshold(rms_values)
            labels = label_segments(rms_values, threshold)
            
            # Update GUI in main thread
            self.root.after(0, lambda: self.compression_completed(results, original_size, labels, threshold))
            
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: self.compression_error(msg))
    
    def compression_completed(self, results, original_size, labels, threshold):
        """Called when compression completes successfully"""
        # Re-enable button
        self.compress_button.config(
            text="Sıkıştır",
            state="normal",
            bg="#4CAF50"
        )
        
        self.is_processing = False
        self.status_var.set("Tamamlandı!")
        
        # Insert results into treeview
        strategies = [
            ('lossless_uniform', 'Kayıpsız - Tekdüze'),
            ('lossless_adaptive', 'Kayıpsız - Adaptif'),
            ('lossy_uniform', 'Kayıplı  - Tekdüze'),
            ('lossy_adaptive', 'Kayıplı  - Adaptif')
        ]
        # Try to add RVQ-GAN strategy if available
        try:
            import rvqgan_compressor
            strategies.append(('rvqgan', 'Kayıplı  - RVQ-GAN'))
        except ImportError:
            pass
        # Try to add Entropy Selection strategy if available
        try:
            import entropy_selector
            strategies.append(('entropy_selection', 'Akıllı   - Entropy Seçici'))
        except ImportError:
            pass
        
        best_ratio = 0
        best_strategy = ""
        
        for key, name in strategies:
            res = results.get(key, {'size': original_size, 'ratio': 0.0, 'snr': None})
            size_str = main_module.utils.format_size(res['size'])
            ratio_str = f"%{res['ratio']*100:.1f}"
            snr_str = "-" if res['snr'] is None else ("∞" if np.isinf(res['snr']) else f"{res['snr']:.2f}")
            
            # Track best compression ratio
            if res['ratio'] > best_ratio:
                best_ratio = res['ratio']
                best_strategy = name
            
            self.tree.insert("", "end", values=(name, size_str, ratio_str, snr_str))
        
        # Update summary
        original_size_str = main_module.utils.format_size(original_size)
        best_ratio_percent = f"{best_ratio*100:.1f}"
        summary_text = f"Orijinal: {original_size_str} | En iyi oran: %{best_ratio_percent} ({best_strategy})"
        
        # Add note for lossless adaptive if it performed worse than uniform
        lossless_uniform_ratio = results.get('lossless_uniform', {}).get('ratio', 0)
        lossless_adaptive_ratio = results.get('lossless_adaptive', {}).get('ratio', 0)
        if lossless_adaptive_ratio > 0 and lossless_uniform_ratio > lossless_adaptive_ratio:
            summary_text += " ℹ️ Müzik dosyasında tekdüze daha iyi performans gösterdi"
        
        self.summary_label.config(text=summary_text)
    
    def compression_error(self, error_msg):
        """Called when compression encounters an error"""
        # Re-enable button
        self.compress_button.config(
            text="Sıkıştır",
            state="normal",
            bg="#4CAF50"
        )
        
        self.is_processing = False
        self.status_var.set("Hata!")
        
        messagebox.showerror("Hata", f"Sıkıştırma sırasında hata oluştu:\n{error_msg}")
    
    def compression_cancelled(self):
        """Called when user cancels file selection"""
        # Re-enable button
        self.compress_button.config(
            text="Sıkıştır",
            state="normal",
            bg="#4CAF50"
        )
        
        self.is_processing = False
        self.status_var.set("İptal edildi")


def main():
    """Main function to run the GUI application"""
    try:
        root = tk.Tk()
        app = AdaptiveAudioCompressionGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Error in GUI: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
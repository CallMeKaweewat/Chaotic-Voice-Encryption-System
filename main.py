import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import numpy as np
import soundfile as sf
import os
import time
import threading
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Crypto.Cipher import ChaCha20, AES
from Crypto.Util import Counter
from Crypto.Hash import SHA256

# --- Configuration ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class ChaosCryptoCore:
    """Core Engine with High-Entropy Quantization"""

    @staticmethod
    def get_initials(password, count=3):
        h = int(SHA256.new(password.encode()).hexdigest(), 16)
        vals = []
        for i in range(count):
            val = ((h >> (i * 12)) % 100000) / 100000.0
            val = 0.1 + (val * 0.8)
            vals.append(val)
        return vals

    @staticmethod
    def safe_byte(val):
        """Deep Decimal Extraction"""
        if not math.isfinite(val): return 0
        try:
            return int((abs(val) * 1000000)) % 256
        except (OverflowError, ValueError):
            return 0

    @staticmethod
    def henon_map(password, length):
        x, y, _ = ChaosCryptoCore.get_initials(password, 3)
        a, b = 1.4, 0.3
        ks = np.zeros(length, dtype=np.uint8)

        # Burn-in
        for _ in range(1000):
            new_x = 1 - (a * (x ** 2)) + y
            y = b * x
            x = new_x
            if abs(x) > 100: x, y = 0.1, 0.1

        # Generation Loop
        for i in range(length):
            new_x = 1 - (a * (x ** 2)) + y
            y = b * x
            x = new_x

            if not math.isfinite(x) or abs(x) > 100.0:
                x, y = 0.1, 0.1

            combined_val = x + y
            ks[i] = ChaosCryptoCore.safe_byte(combined_val)
        return ks

    @staticmethod
    def logistic_map(password, length):
        x = ChaosCryptoCore.get_initials(password, 1)[0]
        r = 3.999
        ks = np.zeros(length, dtype=np.uint8)
        for _ in range(1000):
            x = r * x * (1 - x)
        for i in range(length):
            x = r * x * (1 - x)
            ks[i] = int(x * 1000000) % 256
        return ks

    @staticmethod
    def lorenz_system(password, length):
        x, y, z = ChaosCryptoCore.get_initials(password, 3)
        sigma, rho, beta, dt = 10.0, 28.0, 8.0 / 3.0, 0.01
        ks = np.zeros(length, dtype=np.uint8)
        for _ in range(1000):
            dx = sigma * (y - x) * dt
            dy = (x * (rho - z) - y) * dt
            dz = (x * y - beta * z) * dt
            x += dx;
            y += dy;
            z += dz
        for i in range(length):
            dx = sigma * (y - x) * dt
            dy = (x * (rho - z) - y) * dt
            dz = (x * y - beta * z) * dt
            x += dx;
            y += dy;
            z += dz
            if not math.isfinite(x): x, y, z = 0.1, 0.1, 0.1
            ks[i] = ChaosCryptoCore.safe_byte(abs(x) + abs(z))
        return ks

    @staticmethod
    def chen_system(password, length):
        x, y, z = ChaosCryptoCore.get_initials(password, 3)
        a, b, c, dt = 35.0, 3.0, 28.0, 0.005
        ks = np.zeros(length, dtype=np.uint8)
        for _ in range(1000):
            dx = a * (y - x) * dt
            dy = ((c - a) * x - x * z + c * y) * dt
            dz = (x * y - b * z) * dt
            x += dx;
            y += dy;
            z += dz
        for i in range(length):
            dx = a * (y - x) * dt
            dy = ((c - a) * x - x * z + c * y) * dt
            dz = (x * y - b * z) * dt
            x += dx;
            y += dy;
            z += dz
            if not math.isfinite(x): x, y, z = 0.1, 0.1, 0.1
            ks[i] = ChaosCryptoCore.safe_byte(abs(x) + abs(y))
        return ks

    @staticmethod
    def aes_ctr(password, length):
        key = SHA256.new(password.encode()).digest()
        ctr = Counter.new(64, prefix=key[:8])
        cipher = AES.new(key, AES.MODE_CTR, counter=ctr)
        return np.frombuffer(cipher.encrypt(b'\x00' * length), dtype=np.uint8)

    @staticmethod
    def chacha20(password, length):
        key = SHA256.new(password.encode()).digest()
        cipher = ChaCha20.new(key=key, nonce=key[:12])
        return np.frombuffer(cipher.encrypt(b'\x00' * length), dtype=np.uint8)

    @staticmethod
    def calculate_entropy(data):
        if len(data) == 0: return 0.0
        if data.dtype != np.uint8:
            data = data.view(np.uint8)
        data = data.flatten()
        counts = np.bincount(data, minlength=256)
        probs = counts / len(data)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))


class UltimateCryptoApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Audio Crypto Suite v9.1 (Fixed Layout)")
        self.geometry("1280x900")

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(padx=10, pady=10, fill="both", expand=True)

        self.tab_op = self.tabview.add("🎧 Operation & Visualize")
        self.tab_bench = self.tabview.add("📊 Research/Benchmark")

        self.setup_operation_tab()
        self.setup_benchmark_tab()

    def setup_operation_tab(self):
        # --- Left Panel ---
        ctrl_frame = ctk.CTkFrame(self.tab_op, width=380)
        ctrl_frame.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkLabel(ctrl_frame, text="CONTROL PANEL", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)

        self.file_path = tk.StringVar()
        ctk.CTkButton(ctrl_frame, text="📂 Select Audio File", command=self.browse_file).pack(pady=5, padx=10, fill="x")
        self.lbl_file = ctk.CTkLabel(ctrl_frame, text="No file selected", text_color="gray", wraplength=300)
        self.lbl_file.pack(pady=5)

        ctk.CTkLabel(ctrl_frame, text="Algorithm:").pack(anchor="w", padx=10)
        self.algo_var = ctk.StringVar(value="Hénon Map (2D)")
        algos = ["Hénon Map (2D)", "Logistic Map (1D)", "Lorenz System (3D)", "Chen System (3D)", "ChaCha20", "AES-CTR"]
        ctk.CTkOptionMenu(ctrl_frame, values=algos, variable=self.algo_var).pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(ctrl_frame, text="Password:").pack(anchor="w", padx=10)
        self.entry_pass = ctk.CTkEntry(ctrl_frame, show="*")
        self.entry_pass.pack(fill="x", padx=10, pady=5)

        # --- Button Groups (Fixed for MSB Logic) ---
        ctk.CTkLabel(ctrl_frame, text="--- Standard Mode (High Security) ---", text_color="#aaaaaa").pack(pady=(20, 5))

        btn_grid_std = ctk.CTkFrame(ctrl_frame, fg_color="transparent")
        btn_grid_std.pack(fill="x", padx=5)

        ctk.CTkButton(btn_grid_std, text="🔒 Full Encrypt", fg_color="#c0392b", hover_color="#922b21",
                      command=lambda: self.run_process("encrypt", "full")).pack(side="left", expand=True, padx=2)
        ctk.CTkButton(btn_grid_std, text="🔓 Full Decrypt", fg_color="#27ae60", hover_color="#1e8449",
                      command=lambda: self.run_process("decrypt", "full")).pack(side="right", expand=True, padx=2)

        ctk.CTkLabel(ctrl_frame, text="--- Fast Mode (Selective MSB) ---", text_color="#aaaaaa").pack(pady=(15, 5))

        btn_grid_fast = ctk.CTkFrame(ctrl_frame, fg_color="transparent")
        btn_grid_fast.pack(fill="x", padx=5)

        ctk.CTkButton(btn_grid_fast, text="⚡ MSB Encrypt", fg_color="#d35400", hover_color="#a04000",
                      command=lambda: self.run_process("encrypt", "selective")).pack(side="left", expand=True, padx=2)
        ctk.CTkButton(btn_grid_fast, text="🔋 MSB Decrypt", fg_color="#2980b9", hover_color="#1a5276",
                      command=lambda: self.run_process("decrypt", "selective")).pack(side="right", expand=True, padx=2)

        # --- Log ---
        ctk.CTkLabel(ctrl_frame, text="Status Log:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10,
                                                                                           pady=(20, 5))
        self.op_log = ctk.CTkTextbox(ctrl_frame, font=("Consolas", 12))
        self.op_log.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Right Panel ---
        self.viz_frame = ctk.CTkFrame(self.tab_op)
        self.viz_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(self.viz_frame, text="SIGNAL COMPARISON (Waveform)",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        self.canvas_area = ctk.CTkFrame(self.viz_frame)
        self.canvas_area.pack(fill="both", expand=True, padx=5, pady=5)

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3 *.flac")])
        if path:
            self.file_path.set(path)
            self.lbl_file.configure(text=os.path.basename(path))

    def run_process(self, mode, strategy):
        threading.Thread(target=self.process_thread, args=(mode, strategy), daemon=True).start()

    def process_thread(self, mode, strategy):
        path = self.file_path.get()
        pwd = self.entry_pass.get()
        algo = self.algo_var.get()

        if not path or not pwd:
            self.log_to_box("❌ Error: File and Password required!")
            return

        try:
            start_t = time.time()
            self.log_to_box(f"⏳ Starting {mode.upper()} ({strategy})...")

            # Read File
            data, sr = sf.read(path, dtype='int16')
            # Work on a copy to avoid locking issues
            processed_data = data.copy()
            original_size_mb = os.path.getsize(path) / (1024 * 1024)
            audio_duration = len(data) / sr

            # Get flat view for processing
            flat_view = processed_data.reshape(-1)

            # --- Processing Logic ---
            if strategy == "full":
                # Treat entire data as byte stream
                # view(np.uint8) allows direct byte manipulation without reshaping mess
                byte_view = flat_view.view(np.uint8)
                ks = self.get_keystream(algo, pwd, len(byte_view))

                # XOR in place
                byte_view[:] = np.bitwise_xor(byte_view, ks)

            elif strategy == "selective":
                # View as bytes: [Low, High, Low, High, ...] (Little Endian)
                byte_view = flat_view.view(np.uint8)

                # High bytes (MSB) are at odd indices: 1, 3, 5...
                msb_view = byte_view[1::2]

                ks = self.get_keystream(algo, pwd, len(msb_view))

                # XOR only MSB in place
                # This works for BOTH encrypt and decrypt because XOR is symmetric
                msb_view[:] = np.bitwise_xor(msb_view, ks)

            # --- Saving ---
            tag = algo.split()[0]
            if mode == "encrypt":
                suffix = f"_{tag}_{strategy}_ENC.wav"
            else:
                suffix = f"_{tag}_{strategy}_DEC.wav"

            out_path = os.path.splitext(path)[0] + suffix
            sf.write(out_path, processed_data, sr)

            # Stats
            duration = time.time() - start_t
            if duration == 0: duration = 0.001
            throughput = original_size_mb / duration
            entropy = ChaosCryptoCore.calculate_entropy(processed_data)

            self.generate_detailed_report(
                filename=os.path.basename(path), size_mb=original_size_mb, duration_sec=audio_duration,
                algo=algo, mode=mode, strategy=strategy, time_taken=duration, speed=throughput,
                entropy=entropy, out_file=os.path.basename(out_path)
            )

            self.after(0, lambda: self.plot_waveforms(data, processed_data, mode))

        except Exception as e:
            self.log_to_box(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    def generate_detailed_report(self, filename, size_mb, duration_sec, algo, mode, strategy, time_taken, speed,
                                 entropy, out_file):
        sec_status = "Unknown"
        # Logic for status
        if mode == "encrypt":
            if entropy > 7.9:
                sec_status = "✅ High (Random Noise)"
            elif entropy > 7.0:
                sec_status = "⚠️ Moderate"
            else:
                sec_status = "❌ Low (Pattern Detected)"
        else:
            sec_status = "ℹ️ Restored Data"

        report = f"""
{'=' * 40}
[{mode.upper()} SUCCESS]
{'=' * 40}
📂 File:      {filename}
⚙️ Mode:      {strategy.upper()} - {algo}
🚀 Speed:     {speed:.2f} MB/s ({time_taken:.4f}s)
🔒 Entropy:   {entropy:.5f}
📝 Status:    {sec_status}
💾 Saved:     {out_file}
{'=' * 40}
"""
        self.log_to_box(report)

    def log_to_box(self, text):
        self.op_log.insert("end", text + "\n")
        self.op_log.see("end")

    def plot_waveforms(self, original, processed, mode):
        for widget in self.canvas_area.winfo_children(): widget.destroy()

        # Downsample for performance plotting
        y1 = original.flatten()
        y2 = processed.flatten()
        step = max(1, len(y1) // 5000)
        y1 = y1[::step]
        y2 = y2[::step]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
        fig.patch.set_facecolor('#2b2b2b')

        ax1.plot(y1, color='#2ecc71', lw=0.8)
        ax1.set_title("Input Signal", color='white', fontsize=10)
        ax1.set_facecolor('#212121');
        ax1.tick_params(colors='white')

        color = '#e74c3c' if mode == 'encrypt' else '#3498db'
        title = "Output Signal"
        ax2.plot(y2, color=color, lw=0.8)
        ax2.set_title(title, color='white', fontsize=10)
        ax2.set_facecolor('#212121');
        ax2.tick_params(colors='white')

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_area)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def setup_benchmark_tab(self):
        left = ctk.CTkFrame(self.tab_bench, width=280)
        left.pack(side="left", fill="y", padx=10, pady=10)
        ctk.CTkLabel(left, text="BENCHMARK LAB", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
        self.bench_size = ctk.StringVar(value="1.0")
        ctk.CTkLabel(left, text="Data Size (MB):").pack(pady=5)
        ctk.CTkEntry(left, textvariable=self.bench_size).pack(pady=5)
        ctk.CTkButton(left, text="🚀 RUN BENCHMARK", fg_color="#d35400", command=self.run_benchmark).pack(pady=20,
                                                                                                         fill="x",
                                                                                                         padx=10)
        self.bench_log = ctk.CTkTextbox(left, font=("Consolas", 11))
        self.bench_log.pack(pady=10, padx=5, fill="both", expand=True)
        self.right_panel = ctk.CTkFrame(self.tab_bench)
        self.right_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)

    def run_benchmark(self):
        threading.Thread(target=self.process_benchmark, daemon=True).start()

    def process_benchmark(self):
        try:
            mb = float(self.bench_size.get())
            size_bytes = int(mb * 1024 * 1024)
            pwd = "BENCH_KEY"
            self.bench_log.delete("1.0", "end")
            self.bench_log.insert("end", f"Benchmarking {mb} MB...\n")

            dummy = np.random.randint(0, 256, size_bytes, dtype=np.uint8)
            algos = {
                "Logistic": ChaosCryptoCore.logistic_map,
                "Hénon": ChaosCryptoCore.henon_map,
                "Lorenz": ChaosCryptoCore.lorenz_system,
                "Chen": ChaosCryptoCore.chen_system,
                "ChaCha20": ChaosCryptoCore.chacha20,
                "AES-CTR": ChaosCryptoCore.aes_ctr
            }
            results = []
            for name, func in algos.items():
                self.bench_log.insert("end", f"> {name}...\n")
                start = time.time()
                ks = func(pwd, size_bytes)
                if isinstance(ks, bytes): ks = np.frombuffer(ks, dtype=np.uint8)
                enc = np.bitwise_xor(dummy, ks)
                dur = time.time() - start
                if dur == 0: dur = 0.001
                results.append({"Algo": name, "Speed": mb / dur, "Entropy": ChaosCryptoCore.calculate_entropy(enc)})

            df = pd.DataFrame(results)
            self.bench_log.insert("end", "\nDONE.\n")
            self.after(0, lambda: self.plot_graphs(df))

        except Exception as e:
            self.bench_log.insert("end", f"\nError: {e}")

    def plot_graphs(self, df):
        for widget in self.right_panel.winfo_children(): widget.destroy()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
        fig.patch.set_facecolor('#2b2b2b')

        # Plot Speed
        ax1.barh(df['Algo'], df['Speed'], color='#3498db')
        ax1.set_title('Encryption Speed (MB/s)', color='white')
        ax1.set_facecolor('#2b2b2b');
        ax1.tick_params(colors='white')

        # Plot Entropy
        ax2.barh(df['Algo'], df['Entropy'], color='#e74c3c')
        ax2.set_xlim(7.98, 8.005)  # Zoom in to see small differences
        ax2.axvline(8.0, color='lime', linestyle='--', alpha=0.5)
        ax2.set_title('Entropy (Max=8.0)', color='white')
        ax2.set_facecolor('#2b2b2b');
        ax2.tick_params(colors='white')

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.right_panel)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def get_keystream(self, name, pwd, length):
        if "Hénon" in name: return ChaosCryptoCore.henon_map(pwd, length)
        if "Logistic" in name: return ChaosCryptoCore.logistic_map(pwd, length)
        if "Lorenz" in name: return ChaosCryptoCore.lorenz_system(pwd, length)
        if "Chen" in name: return ChaosCryptoCore.chen_system(pwd, length)
        if "ChaCha20" in name: return ChaosCryptoCore.chacha20(pwd, length)
        if "AES" in name: return ChaosCryptoCore.aes_ctr(pwd, length)
        return np.zeros(length, dtype=np.uint8)


if __name__ == "__main__":
    app = UltimateCryptoApp()
    app.mainloop()
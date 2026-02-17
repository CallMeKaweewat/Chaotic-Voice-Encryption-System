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
    """Core Engine with Crash Protection & Math Logic"""

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
    def safe_byte(val, scale=1.0, offset=0.0):
        if not math.isfinite(val): return 0
        try:
            return int((val + offset) * scale) % 256
        except (OverflowError, ValueError):
            return 0

    @staticmethod
    def henon_map(password, length):
        x, y, _ = ChaosCryptoCore.get_initials(password, 3)
        a, b = 1.4, 0.3
        ks = np.zeros(length, dtype=np.uint8)
        for i in range(length):
            new_x = 1 - (a * (x ** 2)) + y
            y = b * x
            x = new_x
            if not math.isfinite(x) or abs(x) > 100.0: x, y = 0.1, 0.1
            ks[i] = ChaosCryptoCore.safe_byte(x, scale=80, offset=1.5)
        return ks

    @staticmethod
    def logistic_map(password, length):
        x = ChaosCryptoCore.get_initials(password, 1)[0]
        r = 3.999
        ks = np.zeros(length, dtype=np.uint8)
        for i in range(length):
            x = r * x * (1 - x)
            ks[i] = int(x * 255) % 256
        return ks

    @staticmethod
    def lorenz_system(password, length):
        x, y, z = ChaosCryptoCore.get_initials(password, 3)
        sigma, rho, beta, dt = 10.0, 28.0, 8.0 / 3.0, 0.01
        ks = np.zeros(length, dtype=np.uint8)
        for i in range(length):
            dx = sigma * (y - x) * dt
            dy = (x * (rho - z) - y) * dt
            dz = (x * y - beta * z) * dt
            x += dx;
            y += dy;
            z += dz
            if not math.isfinite(x): x, y, z = 0.1, 0.1, 0.1
            ks[i] = ChaosCryptoCore.safe_byte(abs(x), scale=1000)
        return ks

    @staticmethod
    def chen_system(password, length):
        x, y, z = ChaosCryptoCore.get_initials(password, 3)
        a, b, c, dt = 35.0, 3.0, 28.0, 0.005
        ks = np.zeros(length, dtype=np.uint8)
        for i in range(length):
            dx = a * (y - x) * dt
            dy = ((c - a) * x - x * z + c * y) * dt
            dz = (x * y - b * z) * dt
            x += dx;
            y += dy;
            z += dz
            if not math.isfinite(x): x, y, z = 0.1, 0.1, 0.1
            ks[i] = ChaosCryptoCore.safe_byte(abs(x), scale=1000)
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
        counts = np.bincount(data, minlength=256)
        probs = counts / len(data)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    @staticmethod
    def calculate_correlation(original, encrypted):
        limit = min(len(original), 5000)
        if limit == 0: return 0.0
        if np.std(original[:limit]) == 0 or np.std(encrypted[:limit]) == 0: return 0.0
        return np.corrcoef(original[:limit], encrypted[:limit])[0, 1]


class UltimateCryptoApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Audio Crypto Suite v7.0 (Waveform Visualization)")
        self.geometry("1200x900")

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(padx=10, pady=10, fill="both", expand=True)

        self.tab_op = self.tabview.add("🎧 Operation & Visualize")
        self.tab_bench = self.tabview.add("📊 Research/Benchmark")

        self.setup_operation_tab()
        self.setup_benchmark_tab()

    # =========================================================================
    # TAB 1: OPERATION & VISUALIZATION
    # =========================================================================
    def setup_operation_tab(self):
        # --- Left Panel: Controls (Width 350) ---
        ctrl_frame = ctk.CTkFrame(self.tab_op, width=350)
        ctrl_frame.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkLabel(ctrl_frame, text="CONTROL PANEL", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)

        # File
        self.file_path = tk.StringVar()
        ctk.CTkButton(ctrl_frame, text="📂 Select Audio File", command=self.browse_file).pack(pady=5, padx=10, fill="x")
        self.lbl_file = ctk.CTkLabel(ctrl_frame, text="No file selected", text_color="gray", wraplength=300)
        self.lbl_file.pack(pady=5)

        # Settings
        ctk.CTkLabel(ctrl_frame, text="Algorithm:").pack(anchor="w", padx=10)
        self.algo_var = ctk.StringVar(value="Hénon Map (2D)")
        algos = ["Hénon Map (2D)", "Logistic Map (1D)", "Lorenz System (3D)", "Chen System (3D)", "ChaCha20", "AES-CTR"]
        ctk.CTkOptionMenu(ctrl_frame, values=algos, variable=self.algo_var).pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(ctrl_frame, text="Password:").pack(anchor="w", padx=10)
        self.entry_pass = ctk.CTkEntry(ctrl_frame, show="*")
        self.entry_pass.pack(fill="x", padx=10, pady=5)

        # Buttons
        ctk.CTkLabel(ctrl_frame, text="Actions:").pack(anchor="w", padx=10, pady=(20, 5))
        ctk.CTkButton(ctrl_frame, text="🔒 FULL ENCRYPT", fg_color="#c0392b", hover_color="#922b21",
                      command=lambda: self.run_process("encrypt", "full")).pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(ctrl_frame, text="⚡ FAST ENCRYPT (MSB)", fg_color="#d35400", hover_color="#a04000",
                      command=lambda: self.run_process("encrypt", "selective")).pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(ctrl_frame, text="🔓 DECRYPT", fg_color="#27ae60", hover_color="#1e8449",
                      command=lambda: self.run_process("decrypt", "full")).pack(fill="x", padx=10, pady=5)

        # Log
        ctk.CTkLabel(ctrl_frame, text="Status Log:").pack(anchor="w", padx=10, pady=(20, 5))
        self.op_log = ctk.CTkTextbox(ctrl_frame, font=("Consolas", 11))
        self.op_log.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Right Panel: Visualization (Waveforms) ---
        self.viz_frame = ctk.CTkFrame(self.tab_op)
        self.viz_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(self.viz_frame, text="SIGNAL COMPARISON (Waveform)",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        # Placeholder for graph
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
            self.log_op("Error: File and Password required!")
            return

        try:
            start_t = time.time()
            self.log_op(f"Starting {mode.upper()} ({strategy})...")

            # Read File
            data, sr = sf.read(path, dtype='int16')
            flat_data = data.flatten()

            # Processing Logic
            processed_flat = None
            if strategy == "full":
                byte_data = flat_data.tobytes()
                ks = self.get_keystream(algo, pwd, len(byte_data))
                if isinstance(ks, bytes): ks = np.frombuffer(ks, dtype=np.uint8)
                enc_bytes = np.bitwise_xor(np.frombuffer(byte_data, dtype=np.uint8), ks)
                processed_flat = np.frombuffer(enc_bytes.tobytes(), dtype='int16')

            elif strategy == "selective":
                msb = (flat_data >> 8).astype(np.uint8)
                lsb = (flat_data & 0xFF).astype(np.uint8)
                ks = self.get_keystream(algo, pwd, len(msb))
                if isinstance(ks, bytes): ks = np.frombuffer(ks, dtype=np.uint8)
                enc_msb = np.bitwise_xor(msb, ks)
                processed_msb_16 = enc_msb.astype(np.int16) << 8
                processed_flat = processed_msb_16 | lsb.astype(np.int16)

            # Reshape & Save
            final_audio = processed_flat[:len(flat_data)].reshape(data.shape)
            tag = algo.split()[0]
            suffix = f"_{tag}_NOISE.wav" if mode == "encrypt" else f"_{tag}_RESTORED.wav"
            out_path = os.path.splitext(path)[0] + suffix
            sf.write(out_path, final_audio, sr)

            duration = time.time() - start_t

            # Update Log
            self.log_op(f"Done in {duration:.2f}s | Saved: {os.path.basename(out_path)}")

            # Update Graph (Run on main thread)
            self.after(0, lambda: self.plot_waveforms(data, final_audio, mode))

        except Exception as e:
            self.log_op(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def plot_waveforms(self, original, processed, mode):
        # Clear old graph
        for widget in self.canvas_area.winfo_children(): widget.destroy()

        # Handle Stereo (Pick first channel if stereo)
        y1 = original[:, 0] if original.ndim > 1 else original
        y2 = processed[:, 0] if processed.ndim > 1 else processed

        # Downsample for speed (Graphing 10MB of data will freeze UI)
        step = max(1, len(y1) // 5000)
        y1 = y1[::step]
        y2 = y2[::step]

        # Setup Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
        fig.patch.set_facecolor('#2b2b2b')

        # Graph 1: Original
        ax1.plot(y1, color='#2ecc71', lw=0.8)
        ax1.set_title("Original Signal", color='white', fontsize=10)
        ax1.set_facecolor('#212121')
        ax1.tick_params(axis='x', colors='white')
        ax1.tick_params(axis='y', colors='white')
        ax1.grid(True, alpha=0.1)

        # Graph 2: Processed
        color = '#e74c3c' if mode == 'encrypt' else '#3498db'
        title = "Encrypted Signal (Noise)" if mode == 'encrypt' else "Restored Signal"
        ax2.plot(y2, color=color, lw=0.8)
        ax2.set_title(title, color='white', fontsize=10)
        ax2.set_facecolor('#212121')
        ax2.tick_params(axis='x', colors='white')
        ax2.tick_params(axis='y', colors='white')
        ax2.grid(True, alpha=0.1)

        plt.tight_layout()

        # Embed in CustomTkinter
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_area)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def log_op(self, text):
        self.op_log.insert("end", text + "\n")
        self.op_log.see("end")

    # =========================================================================
    # TAB 2: BENCHMARK (Same as before)
    # =========================================================================
    def setup_benchmark_tab(self):
        left = ctk.CTkFrame(self.tab_bench, width=280)
        left.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkLabel(left, text="BENCHMARK LAB", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
        self.bench_size = ctk.StringVar(value="0.5")
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
            self.bench_log.insert("end", "\n" + df.to_string(
                formatters={'Speed': '{:,.2f}'.format, 'Entropy': '{:,.4f}'.format}))
            self.after(0, lambda: self.plot_graphs(df))

        except Exception as e:
            self.bench_log.insert("end", f"\nError: {e}")

    def plot_graphs(self, df):
        for widget in self.right_panel.winfo_children(): widget.destroy()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
        fig.patch.set_facecolor('#2b2b2b')
        ax1.barh(df['Algo'], df['Speed'], color='#3498db')
        ax1.set_title('Speed (MB/s)', color='white')
        ax1.set_facecolor('#2b2b2b');
        ax1.tick_params(colors='white')
        ax2.barh(df['Algo'], df['Entropy'], color='#e74c3c')
        ax2.set_xlim(7.90, 8.01);
        ax2.axvline(8.0, color='lime', linestyle='--')
        ax2.set_title('Entropy', color='white')
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
        return None


if __name__ == "__main__":
    app = UltimateCryptoApp()
    app.mainloop()
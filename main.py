import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import numpy as np
import soundfile as sf
import sounddevice as sd
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
import scipy.stats as stats

# --- Configuration ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class ChaosCryptoCore:
    """Core Engine with High-Entropy Quantization & Analytics"""

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
        for _ in range(1000):
            new_x = 1 - (a * (x ** 2)) + y
            y = b * x
            x = new_x
            if abs(x) > 100: x, y = 0.1, 0.1
        for i in range(length):
            new_x = 1 - (a * (x ** 2)) + y
            y = b * x
            x = new_x
            if not math.isfinite(x) or abs(x) > 100.0: x, y = 0.1, 0.1
            ks[i] = ChaosCryptoCore.safe_byte(x + y)
        return ks

    @staticmethod
    def logistic_map(password, length):
        x = ChaosCryptoCore.get_initials(password, 1)[0]
        r = 3.999
        ks = np.zeros(length, dtype=np.uint8)
        for _ in range(1000): x = r * x * (1 - x)
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
    def calculate_metrics(original, encrypted):
        orig_flat = original.flatten().view(np.uint8)
        enc_flat = encrypted.flatten().view(np.uint8)
        sample_size = min(len(orig_flat), 100000)
        correlation, _ = stats.pearsonr(orig_flat[:sample_size], enc_flat[:sample_size])
        counts = np.bincount(enc_flat, minlength=256)
        probs = counts / len(enc_flat)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        return correlation, entropy

    @classmethod
    def get_algorithms(cls):
        return {
            "Hénon Map (2D)": cls.henon_map,
            "Logistic Map (1D)": cls.logistic_map,
            "Lorenz System (3D)": cls.lorenz_system,
            "Chen System (3D)": cls.chen_system,
            "ChaCha20": cls.chacha20,
            "AES-CTR": cls.aes_ctr
        }


class UltimateCryptoApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Audio Crypto Suite v16.0 (Lossless Integrity Edition)")
        self.geometry("1400x900")

        self.original_audio = None
        self.processed_audio = None
        self.sample_rate = None

        self.stream = None
        self.current_frame = 0
        self.playing_audio = None

        self.latest_benchmark_df = None
        self.latest_benchmark_fig = None

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(padx=10, pady=10, fill="both", expand=True)

        self.tab_op = self.tabview.add("🎧 Operation & Visualize")
        self.tab_bench = self.tabview.add("📊 Research/Benchmark")

        self.setup_operation_tab()
        self.setup_benchmark_tab()

    def setup_operation_tab(self):
        ctrl_frame = ctk.CTkFrame(self.tab_op, width=380)
        ctrl_frame.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkLabel(ctrl_frame, text="CONTROL PANEL", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)

        self.file_path = tk.StringVar()
        ctk.CTkButton(ctrl_frame, text="📂 Select Audio File", command=self.browse_file).pack(pady=5, padx=10, fill="x")
        self.lbl_file = ctk.CTkLabel(ctrl_frame, text="No file selected", text_color="gray", wraplength=300)
        self.lbl_file.pack(pady=5)

        algo_names = list(ChaosCryptoCore.get_algorithms().keys())
        ctk.CTkLabel(ctrl_frame, text="Algorithm:").pack(anchor="w", padx=10)
        self.algo_var = ctk.StringVar(value=algo_names[0])
        ctk.CTkOptionMenu(ctrl_frame, values=algo_names, variable=self.algo_var).pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(ctrl_frame, text="Password:").pack(anchor="w", padx=10)
        self.entry_pass = ctk.CTkEntry(ctrl_frame, show="*")
        self.entry_pass.pack(fill="x", padx=10, pady=5)

        # --- [NEW] Format Selector ---
        ctk.CTkLabel(ctrl_frame, text="Output Format (Lossless):").pack(anchor="w", padx=10, pady=(5, 0))
        self.format_var = ctk.StringVar(value=".wav")
        ctk.CTkOptionMenu(ctrl_frame, values=[".wav (Fast/Large)", ".flac (Slower/Smaller)"],
                          variable=self.format_var).pack(fill="x", padx=10, pady=5)

        self.progress = ctk.CTkProgressBar(ctrl_frame)
        self.progress.pack(fill="x", padx=10, pady=(15, 5))
        self.progress.set(0)

        btn_grid_std = ctk.CTkFrame(ctrl_frame, fg_color="transparent")
        btn_grid_std.pack(fill="x", padx=5, pady=(5, 0))
        ctk.CTkButton(btn_grid_std, text="🔒 Full Enc", fg_color="#c0392b",
                      command=lambda: self.run_process("encrypt", "full")).pack(side="left", expand=True, padx=2)
        ctk.CTkButton(btn_grid_std, text="🔓 Full Dec", fg_color="#27ae60",
                      command=lambda: self.run_process("decrypt", "full")).pack(side="right", expand=True, padx=2)

        btn_grid_msb = ctk.CTkFrame(ctrl_frame, fg_color="transparent")
        btn_grid_msb.pack(fill="x", padx=5, pady=(5, 0))
        ctk.CTkButton(btn_grid_msb, text="⚡ MSB Enc", fg_color="#d35400",
                      command=lambda: self.run_process("encrypt", "selective")).pack(side="left", expand=True, padx=2)
        ctk.CTkButton(btn_grid_msb, text="🔋 MSB Dec", fg_color="#2980b9",
                      command=lambda: self.run_process("decrypt", "selective")).pack(side="right", expand=True, padx=2)

        ctk.CTkLabel(ctrl_frame, text="Volume Control:").pack(anchor="w", padx=10, pady=(15, 0))
        self.vol_slider = ctk.CTkSlider(ctrl_frame, from_=0.0, to=1.0)
        self.vol_slider.set(0.5)
        self.vol_slider.pack(fill="x", padx=10, pady=(0, 10))

        btn_grid_play = ctk.CTkFrame(ctrl_frame, fg_color="transparent")
        btn_grid_play.pack(fill="x", padx=5)
        self.btn_play_orig = ctk.CTkButton(btn_grid_play, text="▶ Original", fg_color="#2ecc71",
                                           command=self.play_original, state="disabled")
        self.btn_play_orig.pack(side="left", expand=True, padx=2)
        self.btn_play_proc = ctk.CTkButton(btn_grid_play, text="▶ Processed", fg_color="#e74c3c",
                                           command=self.play_processed, state="disabled")
        self.btn_play_proc.pack(side="right", expand=True, padx=2)

        self.btn_stop = ctk.CTkButton(ctrl_frame, text="⏹ Stop Audio", fg_color="#7f8c8d", command=self.stop_audio,
                                      state="disabled")
        self.btn_stop.pack(fill="x", padx=10, pady=5)

        self.op_log = ctk.CTkTextbox(ctrl_frame, font=("Consolas", 12))
        self.op_log.pack(fill="both", expand=True, padx=10, pady=10)

        self.viz_frame = ctk.CTkFrame(self.tab_op)
        self.viz_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        self.canvas_area = ctk.CTkFrame(self.viz_frame)
        self.canvas_area.pack(fill="both", expand=True, padx=5, pady=5)

    def setup_benchmark_tab(self):
        left = ctk.CTkFrame(self.tab_bench, width=320)
        left.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkLabel(left, text="🔬 BENCHMARK LAB", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)

        self.bench_file_path = tk.StringVar()
        ctk.CTkButton(left, text="📂 Select File", fg_color="#34495e", command=self.browse_bench_file).pack(pady=5,
                                                                                                           padx=10,
                                                                                                           fill="x")
        self.lbl_bench_file = ctk.CTkLabel(left, text="No file selected (dummy)", text_color="gray")
        self.lbl_bench_file.pack(pady=5)

        ctk.CTkLabel(left, text="Select Algorithms:").pack(anchor="w", padx=10, pady=(10, 0))

        self.bench_algo_vars = {}
        algo_names = list(ChaosCryptoCore.get_algorithms().keys())

        algo_frame = ctk.CTkFrame(left, fg_color="#2b2b2b")
        algo_frame.pack(fill="x", padx=10, pady=5)

        for algo in algo_names:
            var = tk.BooleanVar(value=True)
            chk = ctk.CTkCheckBox(algo_frame, text=algo, variable=var, checkbox_height=18, checkbox_width=18)
            chk.pack(anchor="w", padx=10, pady=3)
            self.bench_algo_vars[algo] = var

        btn_frame = ctk.CTkFrame(left, fg_color="transparent")
        btn_frame.pack(pady=10, fill="x", padx=10)

        ctk.CTkButton(btn_frame, text="🔒 RUN FULL BENCHMARK", fg_color="#c0392b",
                      command=lambda: self.run_benchmark("full")).pack(side="top", fill="x", pady=5)
        ctk.CTkButton(btn_frame, text="⚡ RUN MSB BENCHMARK", fg_color="#d35400",
                      command=lambda: self.run_benchmark("msb")).pack(side="top", fill="x", pady=5)
        ctk.CTkButton(btn_frame, text="💾 EXPORT DATA & GRAPHS", fg_color="#27ae60", hover_color="#2ecc71",
                      command=self.export_results).pack(side="top", fill="x", pady=(15, 5))

        self.bench_log = ctk.CTkTextbox(left, font=("Consolas", 11))
        self.bench_log.pack(pady=10, padx=5, fill="both", expand=True)

        self.bench_viz = ctk.CTkFrame(self.tab_bench)
        self.bench_viz.pack(side="right", fill="both", expand=True, padx=10, pady=10)

    def export_results(self):
        if self.latest_benchmark_df is None or self.latest_benchmark_df.empty:
            messagebox.showwarning("No Data", "Please run a benchmark first before exporting!")
            return

        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")],
                                                title="Save Benchmark Results")
        if filepath:
            try:
                self.latest_benchmark_df.to_csv(filepath, index=False, encoding='utf-8')
                if self.latest_benchmark_fig is not None:
                    img_path = os.path.splitext(filepath)[0] + ".png"
                    self.latest_benchmark_fig.savefig(img_path, facecolor='#2b2b2b', bbox_inches='tight')
                    messagebox.showinfo("Success",
                                        f"Exported Successfully!\n\nData: {os.path.basename(filepath)}\nGraph: {os.path.basename(img_path)}")
                else:
                    messagebox.showinfo("Success", f"Data exported successfully to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file:\n{str(e)}")

    def play_audio_safe(self, audio_data):
        self.stop_audio()
        if audio_data is not None and self.sample_rate:
            self.current_frame = 0
            self.playing_audio = audio_data.astype(np.float32) / 32768.0
            channels = 1 if audio_data.ndim == 1 else audio_data.shape[1]

            def callback(outdata, frames, time, status):
                vol = self.vol_slider.get()
                chunksize = min(len(self.playing_audio) - self.current_frame, frames)
                if chunksize == 0: raise sd.CallbackStop()
                data_chunk = self.playing_audio[self.current_frame:self.current_frame + chunksize] * vol
                if data_chunk.ndim == 1: data_chunk = data_chunk.reshape(-1, 1)
                outdata[:chunksize] = data_chunk
                if chunksize < frames:
                    outdata[chunksize:] = 0
                    raise sd.CallbackStop()
                self.current_frame += chunksize

            self.stream = sd.OutputStream(samplerate=self.sample_rate, channels=channels, callback=callback)
            self.stream.start()

    def play_original(self):
        self.play_audio_safe(self.original_audio)

    def play_processed(self):
        self.play_audio_safe(self.processed_audio)

    def stop_audio(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def enable_playback_buttons(self):
        self.btn_play_orig.configure(state="normal")
        self.btn_play_proc.configure(state="normal")
        self.btn_stop.configure(state="normal")

    def update_progress(self, value):
        self.progress.set(value)

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg")])
        if path: self.file_path.set(path); self.lbl_file.configure(text=os.path.basename(path))

    def browse_bench_file(self):
        path = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg")])
        if path: self.bench_file_path.set(path); self.lbl_bench_file.configure(text=os.path.basename(path),
                                                                               text_color="white")

    def get_keystream(self, name, pwd, length):
        algos = ChaosCryptoCore.get_algorithms()
        if name in algos: return algos[name](pwd, length)
        return np.zeros(length, dtype=np.uint8)

    def run_process(self, mode, strategy):
        threading.Thread(target=self.process_thread, args=(mode, strategy), daemon=True).start()

    def process_thread(self, mode, strategy):
        path, pwd, algo = self.file_path.get(), self.entry_pass.get(), self.algo_var.get()
        if not path or not pwd: return
        try:
            filename = os.path.basename(path)
            self.after(0, lambda: self.update_progress(0.0))
            self.after(0, lambda: self.op_log.insert("end",
                                                     f"\n--- Starting {mode.upper()} ({strategy.upper()}) ---\n🎯 Target: {filename}\n⚙️ Algo:   {algo}\n"))

            io_start = time.time()
            data, sr = sf.read(path, dtype='int16')
            self.after(0, lambda: self.update_progress(0.2))
            io_read_time = time.time() - io_start

            processed_data = data.copy()
            flat_data = processed_data.reshape(-1)
            size_mb = (len(flat_data) * 2) / (1024 * 1024)
            self.after(0, lambda: self.op_log.insert("end",
                                                     f"📊 Size:   {size_mb:.4f} MB\n📂 Read:   {io_read_time:.4f} s\n"))

            enc_start = time.time()
            self.after(0, lambda: self.update_progress(0.4))

            if strategy == "full":
                byte_view = flat_data.view(np.uint8)
                ks = self.get_keystream(algo, pwd, len(byte_view))
                self.after(0, lambda: self.update_progress(0.7))
                byte_view[:] = np.bitwise_xor(byte_view, ks[:len(byte_view)])
            else:
                ks = self.get_keystream(algo, pwd, len(flat_data))
                self.after(0, lambda: self.update_progress(0.7))
                mask = ks.astype(np.int16) << 8
                flat_data[:] = np.bitwise_xor(flat_data, mask)

            enc_time = time.time() - enc_start
            enc_speed = size_mb / enc_time if enc_time > 0 else 0
            self.after(0, lambda: self.update_progress(0.9))
            self.after(0, lambda: self.op_log.insert("end", f"🔐 Calc:   {enc_time:.4f} s ({enc_speed:.2f} MB/s)\n"))

            write_start = time.time()

            # --- [NEW] ดึงนามสกุลที่เลือกมาต่อท้ายไฟล์ ---
            out_ext = self.format_var.get().split(" ")[0]  # แยกเอาเฉพาะคำว่า .wav หรือ .flac
            out_path = os.path.splitext(path)[0] + f"_{strategy}_{mode}{out_ext}"
            sf.write(out_path, processed_data, sr)

            write_time = time.time() - write_start

            self.after(0, lambda: self.update_progress(1.0))

            total_time = io_read_time + enc_time + write_time
            total_speed = size_mb / total_time if total_time > 0 else 0

            self.original_audio = data
            self.processed_audio = processed_data
            self.sample_rate = sr

            self.after(0, lambda: self.op_log.insert("end",
                                                     f"💾 Write ({out_ext}): {write_time:.4f} s\n⚡ Total:  {total_speed:.2f} MB/s\n✅ Done in {total_time:.4f} s\n\n"))
            self.after(0, lambda: self.plot_waveforms(data, processed_data, mode))
            self.after(0, self.enable_playback_buttons)
            self.after(1000, lambda: self.update_progress(0.0))

        except Exception as e:
            self.after(0, lambda: self.update_progress(0.0))
            self.after(0, lambda err=e: self.op_log.insert("end", f"❌ Error: {err}\n"))

    def run_benchmark(self, mode):
        threading.Thread(target=self.process_benchmark, args=(mode,), daemon=True).start()

    def process_benchmark(self, mode):
        try:
            algos_to_test = [algo for algo, var in self.bench_algo_vars.items() if var.get()]
            if not algos_to_test:
                self.after(0, lambda: messagebox.showwarning("Warning",
                                                             "Please select at least one algorithm to benchmark!"))
                return

            path = self.bench_file_path.get()
            io_start = time.time()
            if path and os.path.exists(path):
                data_raw, _ = sf.read(path, dtype='int16')
                test_data = data_raw.flatten()
            else:
                test_data = np.random.randint(-32768, 32767, 1024 * 512, dtype=np.int16)
            io_time = time.time() - io_start

            size_mb = (len(test_data) * 2) / (1024 * 1024)
            mode_name = "FULL MODE" if mode == "full" else "MSB MODE"

            self.after(0, lambda: self.bench_log.delete("1.0", "end"))
            self.after(0, lambda: self.bench_log.insert("end",
                                                        f"🚀 Analyzing {size_mb:.4f} MB [{mode_name}]\n📂 I/O Load: {io_time:.4f} s\n" + "-" * 40 + "\n"))

            results = []
            total_start_time = time.time()

            for name in algos_to_test:
                self.after(0, lambda n=name: self.bench_log.insert("end", f"Testing [{n}]...\n"))
                enc_start = time.time()
                enc_data = test_data.copy()

                if mode == "full":
                    byte_view = enc_data.view(np.uint8)
                    ks = self.get_keystream(name, "KEY", len(byte_view))
                    byte_view[:] = np.bitwise_xor(byte_view, ks[:len(byte_view)])
                else:
                    ks = self.get_keystream(name, "KEY", len(enc_data))
                    mask = ks.astype(np.int16) << 8
                    enc_data[:] = np.bitwise_xor(enc_data, mask)

                enc_time = time.time() - enc_start
                speed = size_mb / enc_time if enc_time > 0 else 0
                corr, ent = ChaosCryptoCore.calculate_metrics(test_data, enc_data)

                self.after(0, lambda
                    text=f"  ▶ Enc: {enc_time:.4f}s | Speed: {speed:6.1f} MB/s\n  ▶ Ent: {ent:.4f}  | Corr: {abs(corr):.4f}\n\n": self.bench_log.insert(
                    "end", text))
                results.append(
                    {"Mode": mode_name, "Algo": name, "Speed_MBps": round(speed, 2), "Entropy": round(ent, 5),
                     "Correlation": round(abs(corr), 5)})

            total_duration = time.time() - total_start_time
            df = pd.DataFrame(results)
            self.latest_benchmark_df = df

            self.after(0, lambda: self.bench_log.insert("end",
                                                        f"✅ Benchmark Complete in {total_duration:.2f} s!\n💡 You can now click 'EXPORT DATA & GRAPHS' to save.\n"))
            self.after(0, lambda m=mode: self.plot_bench_graphs(df, m))

        except Exception as e:
            self.after(0, lambda err=e: self.bench_log.insert("end", f"❌ Error: {err}\n"))

    def plot_bench_graphs(self, df, mode):
        for widget in self.bench_viz.winfo_children(): widget.destroy()
        if df.empty: return

        fig = plt.figure(figsize=(9, 11), facecolor='#2b2b2b')
        color_speed = '#3498db' if mode == 'full' else '#2ecc71'
        title_suffix = "(FULL)" if mode == 'full' else "(MSB)"

        ax1 = fig.add_subplot(311, facecolor='#2b2b2b')
        ax1.bar(df['Algo'], df['Speed_MBps'], color=color_speed)
        ax1.set_title(f'Encryption Speed (MB/s) {title_suffix}', color='white')
        ax1.tick_params(colors='white')

        ax2 = fig.add_subplot(312, facecolor='#2b2b2b')
        ax2.plot(df['Algo'], df['Entropy'], marker='o', color='#e74c3c')
        ax2.set_title(f'Overall File Entropy {title_suffix}', color='white')

        min_ent = min(df['Entropy'])
        max_ent = max(df['Entropy'])
        padding = 0.1 if max_ent - min_ent > 0 else 0.5
        ax2.set_ylim(min_ent - padding, max_ent + padding)
        ax2.tick_params(colors='white')

        ax3 = fig.add_subplot(313, facecolor='#2b2b2b')
        ax3.bar(df['Algo'], df['Correlation'], color='#9b59b6')
        ax3.set_title(f'Pearson Correlation {title_suffix}', color='white')
        ax3.tick_params(colors='white')

        plt.tight_layout()
        self.latest_benchmark_fig = fig

        canvas = FigureCanvasTkAgg(fig, master=self.bench_viz)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

    def plot_waveforms(self, original, processed, mode):
        for widget in self.canvas_area.winfo_children(): widget.destroy()
        y1, y2 = original.flatten(), processed.flatten()
        step = max(1, len(y1) // 5000)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), facecolor='#2b2b2b')

        ax1.plot(y1[::step], color='#2ecc71', lw=0.7)
        ax1.set_facecolor('#1a1a1a')
        ax1.set_title("Original Waveform", color="white")

        ax2.plot(y2[::step], color='#e74c3c' if mode == 'encrypt' else '#3498db', lw=0.7)
        ax2.set_facecolor('#1a1a1a')
        ax2.set_title("Processed Waveform", color="white")

        ax1.tick_params(colors='white')
        ax2.tick_params(colors='white')

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_area)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()


if __name__ == "__main__":
    app = UltimateCryptoApp()
    app.mainloop()
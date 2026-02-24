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

    # --- Algorithms ---
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

    # --- Metrics ---
    @staticmethod
    def calculate_metrics(original, encrypted):
        orig_flat = original.flatten().view(np.uint8)
        enc_flat = encrypted.flatten().view(np.uint8)
        sample_size = min(len(orig_flat), 100000)

        # ป้องกัน error กรณีขนาดความยาวไม่เท่ากัน
        correlation, _ = stats.pearsonr(orig_flat[:sample_size], enc_flat[:sample_size])

        # Entropy
        counts = np.bincount(enc_flat, minlength=256)
        probs = counts / len(enc_flat)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))

        return correlation, entropy


class UltimateCryptoApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Audio Crypto Suite v10.0 (Robust MSB Edition)")
        self.geometry("1400x900")

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

        ctk.CTkLabel(ctrl_frame, text="Algorithm:").pack(anchor="w", padx=10)
        self.algo_var = ctk.StringVar(value="Hénon Map (2D)")
        algos = ["Hénon Map (2D)", "Logistic Map (1D)", "Lorenz System (3D)", "Chen System (3D)", "ChaCha20", "AES-CTR"]
        ctk.CTkOptionMenu(ctrl_frame, values=algos, variable=self.algo_var).pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(ctrl_frame, text="Password:").pack(anchor="w", padx=10)
        self.entry_pass = ctk.CTkEntry(ctrl_frame, show="*")
        self.entry_pass.pack(fill="x", padx=10, pady=5)

        # --- Standard Buttons ---
        ctk.CTkLabel(ctrl_frame, text="--- Standard Mode (Full) ---", text_color="#aaaaaa").pack(pady=(20, 5))
        btn_grid_std = ctk.CTkFrame(ctrl_frame, fg_color="transparent")
        btn_grid_std.pack(fill="x", padx=5)
        ctk.CTkButton(btn_grid_std, text="🔒 Full Enc", fg_color="#c0392b",
                      command=lambda: self.run_process("encrypt", "full")).pack(side="left", expand=True, padx=2)
        ctk.CTkButton(btn_grid_std, text="🔓 Full Dec", fg_color="#27ae60",
                      command=lambda: self.run_process("decrypt", "full")).pack(side="right", expand=True, padx=2)

        # --- MSB Buttons ---
        ctk.CTkLabel(ctrl_frame, text="--- Fast Mode (MSB Only) ---", text_color="#aaaaaa").pack(pady=(15, 5))
        btn_grid_msb = ctk.CTkFrame(ctrl_frame, fg_color="transparent")
        btn_grid_msb.pack(fill="x", padx=5)
        ctk.CTkButton(btn_grid_msb, text="⚡ MSB Enc", fg_color="#d35400",
                      command=lambda: self.run_process("encrypt", "selective")).pack(side="left", expand=True, padx=2)
        ctk.CTkButton(btn_grid_msb, text="🔋 MSB Dec", fg_color="#2980b9",
                      command=lambda: self.run_process("decrypt", "selective")).pack(side="right", expand=True, padx=2)

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
        ctk.CTkButton(left, text="📂 Select File for Testing", fg_color="#34495e", command=self.browse_bench_file).pack(
            pady=5, padx=10, fill="x")
        self.lbl_bench_file = ctk.CTkLabel(left, text="No file selected (using dummy)", text_color="gray")
        self.lbl_bench_file.pack(pady=5)

        # --- สร้างปุ่มแยกโหมด ---
        btn_frame = ctk.CTkFrame(left, fg_color="transparent")
        btn_frame.pack(pady=15, fill="x", padx=10)

        ctk.CTkButton(btn_frame, text="🔒 RUN FULL BENCHMARK", fg_color="#c0392b",
                      command=lambda: self.run_benchmark("full")).pack(side="top", fill="x", pady=5)

        ctk.CTkButton(btn_frame, text="⚡ RUN MSB BENCHMARK", fg_color="#d35400",
                      command=lambda: self.run_benchmark("msb")).pack(side="top", fill="x", pady=5)

        self.bench_log = ctk.CTkTextbox(left, font=("Consolas", 11))
        self.bench_log.pack(pady=10, padx=5, fill="both", expand=True)

        self.bench_viz = ctk.CTkFrame(self.tab_bench)
        self.bench_viz.pack(side="right", fill="both", expand=True, padx=10, pady=10)

    # --- Core Logic ---
    def browse_file(self):
        # เพิ่ม *.ogg ในตัวกรองชนิดไฟล์
        path = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg")])
        if path: self.file_path.set(path); self.lbl_file.configure(text=os.path.basename(path))

    def browse_bench_file(self):
        # เพิ่ม *.ogg ในตัวกรองชนิดไฟล์สำหรับหน้า Benchmark
        path = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg")])
        if path: self.bench_file_path.set(path); self.lbl_bench_file.configure(text=os.path.basename(path),
                                                                               text_color="white")

    def get_keystream(self, name, pwd, length):
        if "Hénon" in name: return ChaosCryptoCore.henon_map(pwd, length)
        if "Logistic" in name: return ChaosCryptoCore.logistic_map(pwd, length)
        if "Lorenz" in name: return ChaosCryptoCore.lorenz_system(pwd, length)
        if "Chen" in name: return ChaosCryptoCore.chen_system(pwd, length)
        if "ChaCha20" in name: return ChaosCryptoCore.chacha20(pwd, length)
        if "AES" in name: return ChaosCryptoCore.aes_ctr(pwd, length)
        return np.zeros(length, dtype=np.uint8)

    def run_process(self, mode, strategy):
        threading.Thread(target=self.process_thread, args=(mode, strategy), daemon=True).start()

    def process_thread(self, mode, strategy):
        path, pwd, algo = self.file_path.get(), self.entry_pass.get(), self.algo_var.get()
        if not path or not pwd: return
        try:
            filename = os.path.basename(path)

            # Header Log
            msg_start = f"\n--- Starting {mode.upper()} ({strategy.upper()}) ---\n🎯 Target: {filename}\n⚙️ Algo:   {algo}\n"
            self.after(0, lambda m=msg_start: self.op_log.insert("end", m))

            # 1. จับเวลา File I/O (Read)
            io_start = time.time()
            data, sr = sf.read(path, dtype='int16')
            io_read_time = time.time() - io_start

            processed_data = data.copy()
            flat_data = processed_data.reshape(-1)  # 1D int16 array

            # คำนวณขนาดไฟล์ (MB) (int16 มีขนาด 2 bytes ต่อค่า)
            size_mb = (len(flat_data) * 2) / (1024 * 1024)
            msg_read = f"📊 Size:   {size_mb:.4f} MB\n📂 Read:   {io_read_time:.4f} s\n"
            self.after(0, lambda m=msg_read: self.op_log.insert("end", m))

            # 2. จับเวลา Encryption / Calculation
            enc_start = time.time()

            if strategy == "full":
                byte_view = flat_data.view(np.uint8)
                ks = self.get_keystream(algo, pwd, len(byte_view))
                byte_view[:] = np.bitwise_xor(byte_view, ks[:len(byte_view)])
            else:
                # MSB Mode: ใช้ Bitwise Shift ดันรหัสขึ้นไปประมวลผลเฉพาะ 8 บิตบน (เสถียรสุด 100%)
                ks = self.get_keystream(algo, pwd, len(flat_data))
                mask = ks.astype(np.int16) << 8
                flat_data[:] = np.bitwise_xor(flat_data, mask)

            enc_time = time.time() - enc_start

            # ความเร็วช่วงประมวลผลสมการล้วนๆ
            enc_speed = size_mb / enc_time if enc_time > 0 else 0
            msg_enc = f"🔐 Calc:   {enc_time:.4f} s ({enc_speed:.2f} MB/s)\n"
            self.after(0, lambda m=msg_enc: self.op_log.insert("end", m))

            # 3. จับเวลา File I/O (Write)
            write_start = time.time()
            out_path = os.path.splitext(path)[0] + f"_{strategy}_{mode}.wav"
            sf.write(out_path, processed_data, sr)
            write_time = time.time() - write_start

            # สรุปเวลารวม และความเร็วจริง (Total Real Speed)
            total_time = io_read_time + enc_time + write_time
            total_speed = size_mb / total_time if total_time > 0 else 0

            msg_write = f"💾 Write:  {write_time:.4f} s\n"
            msg_done = f"⚡ Total Real Speed: {total_speed:.2f} MB/s\n✅ Done in {total_time:.4f} s\n\n"

            self.after(0, lambda m=msg_write: self.op_log.insert("end", m))
            self.after(0, lambda m=msg_done: self.op_log.insert("end", m))

            self.after(0, lambda: self.plot_waveforms(data, processed_data, mode))
        except Exception as e:
            self.after(0, lambda err=e: self.op_log.insert("end", f"❌ Error: {err}\n"))

    def run_benchmark(self, mode):
        threading.Thread(target=self.process_benchmark, args=(mode,), daemon=True).start()

    def process_benchmark(self, mode):
        try:
            path = self.bench_file_path.get()

            # --- 1. จับเวลาโหลดไฟล์เข้า Memory (I/O & Decoding) ---
            io_start = time.time()
            if path and os.path.exists(path):
                data_raw, _ = sf.read(path, dtype='int16')
                test_data = data_raw.flatten()  # ใช้โครงสร้าง int16 เต็มรูปแบบ
                data_source = "Audio File"
            else:
                test_data = np.random.randint(-32768, 32767, 1024 * 512, dtype=np.int16)
                data_source = "Dummy Data"
            io_time = time.time() - io_start

            size_mb = (len(test_data) * 2) / (1024 * 1024)
            mode_name = "FULL MODE" if mode == "full" else "MSB MODE"

            self.after(0, lambda: self.bench_log.delete("1.0", "end"))
            self.after(0, lambda: self.bench_log.insert("end", f"🚀 Analyzing {size_mb:.4f} MB [{mode_name}]\n"))
            self.after(0, lambda: self.bench_log.insert("end", f"📂 I/O Load Time ({data_source}): {io_time:.4f} s\n"))
            self.after(0, lambda: self.bench_log.insert("end", "-" * 55 + "\n"))

            algos = ["Logistic", "Hénon", "Lorenz", "Chen", "ChaCha20", "AES-CTR"]
            results = []

            total_start_time = time.time()

            for name in algos:
                self.after(0, lambda n=name: self.bench_log.insert("end", f"Testing [{n}]...\n"))

                # --- 2. จับเวลาคำนวณ Chaos และเข้ารหัส (Encryption Time) ---
                enc_start = time.time()
                enc_data = test_data.copy()

                if mode == "full":
                    byte_view = enc_data.view(np.uint8)
                    ks = self.get_keystream(name, "KEY", len(byte_view))
                    byte_view[:] = np.bitwise_xor(byte_view, ks[:len(byte_view)])
                else:  # MSB mode
                    ks = self.get_keystream(name, "KEY", len(enc_data))
                    mask = ks.astype(np.int16) << 8
                    enc_data[:] = np.bitwise_xor(enc_data, mask)

                enc_time = time.time() - enc_start

                speed = size_mb / enc_time if enc_time > 0 else 0
                corr, ent = ChaosCryptoCore.calculate_metrics(test_data, enc_data)

                log_text = f"  ▶ Enc Time: {enc_time:.4f} s | Speed: {speed:8.2f} MB/s | Ent: {ent:.4f} | Corr: {abs(corr):.4f}\n\n"
                self.after(0, lambda text=log_text: self.bench_log.insert("end", text))

                results.append({"Algo": name, "Speed": speed, "Entropy": ent, "Correlation": abs(corr)})

            total_duration = time.time() - total_start_time

            df = pd.DataFrame(results)
            self.after(0, lambda duration=total_duration: self.bench_log.insert("end",
                                                                                f"✅ Benchmark Complete in {duration:.2f} seconds!\n"))
            self.after(0, lambda m=mode: self.plot_bench_graphs(df, m))

        except Exception as e:
            self.after(0, lambda err=e: self.bench_log.insert("end", f"❌ Error: {err}\n"))

    def plot_bench_graphs(self, df, mode):
        for widget in self.bench_viz.winfo_children(): widget.destroy()
        fig = plt.figure(figsize=(9, 11), facecolor='#2b2b2b')

        color_speed = '#3498db' if mode == 'full' else '#2ecc71'
        title_suffix = "(FULL)" if mode == 'full' else "(MSB)"

        ax1 = fig.add_subplot(311, facecolor='#2b2b2b')
        ax1.bar(df['Algo'], df['Speed'], color=color_speed)
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
        ax1.set_title("Original", color="white")
        ax2.plot(y2[::step], color='#e74c3c' if mode == 'encrypt' else '#3498db', lw=0.7)
        ax2.set_facecolor('#1a1a1a')
        ax2.set_title("Processed", color="white")
        plt.tight_layout()
        FigureCanvasTkAgg(fig, master=self.canvas_area).get_tk_widget().pack(fill="both", expand=True)


if __name__ == "__main__":
    app = UltimateCryptoApp()
    app.mainloop()
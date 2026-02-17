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
ctk.set_default_color_theme("green")


class ChaosCryptoCore:
    """
    Core Engine: รวมอัลกอริทึม Chaos ที่มีระบบป้องกัน Crash (Overflow Protection)
    และเครื่องมือคำนวณทางสถิติสำหรับงานวิจัย
    """

    @staticmethod
    def get_initials(password, count=3):
        # Generate stable initial conditions from password hash
        h = int(SHA256.new(password.encode()).hexdigest(), 16)
        vals = []
        for i in range(count):
            val = ((h >> (i * 12)) % 100000) / 100000.0
            # Normalize to safe range 0.1 - 0.9
            val = 0.1 + (val * 0.8)
            vals.append(val)
        return vals

    @staticmethod
    def safe_byte(val, scale=1.0, offset=0.0):
        """
        [CRITICAL FIX] ป้องกัน Error: float infinity to integer
        ถ้าค่าเป็น Infinity หรือ NaN ให้คืนค่า 0 ทันที เพื่อไม่ให้โปรแกรมเด้ง
        """
        if not math.isfinite(val): return 0
        try:
            return int((val + offset) * scale) % 256
        except (OverflowError, ValueError):
            return 0

    # --- Algorithms ---
    @staticmethod
    def henon_map(password, length):
        """Hénon Map (2D) - Stabilized"""
        x, y, _ = ChaosCryptoCore.get_initials(password, 3)
        a, b = 1.4, 0.3
        ks = np.zeros(length, dtype=np.uint8)

        for i in range(length):
            new_x = 1 - (a * (x ** 2)) + y
            y = b * x
            x = new_x

            # Anti-Divergence Check (ถ้าค่าพุ่งเกิน 100 ให้ Reset)
            if not math.isfinite(x) or abs(x) > 100.0:
                x, y = 0.1, 0.1

            ks[i] = ChaosCryptoCore.safe_byte(x, scale=80, offset=1.5)
        return ks

    @staticmethod
    def logistic_map(password, length):
        """Logistic Map (1D)"""
        x = ChaosCryptoCore.get_initials(password, 1)[0]
        r = 3.999
        ks = np.zeros(length, dtype=np.uint8)
        for i in range(length):
            x = r * x * (1 - x)
            ks[i] = int(x * 255) % 256
        return ks

    @staticmethod
    def lorenz_system(password, length):
        """Lorenz System (3D)"""
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
        """Chen System (3D)"""
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

    # --- Metrics ---
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
        self.title("Ultimate Audio Crypto Suite v6.0 (Selective + Benchmark)")
        self.geometry("1100x850")

        # Tab View
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(padx=20, pady=10, fill="both", expand=True)

        self.tab_op = self.tabview.add("🎧 Operation Mode")
        self.tab_bench = self.tabview.add("📊 Research/Benchmark")

        self.setup_operation_tab()
        self.setup_benchmark_tab()

    # =========================================================================
    # TAB 1: OPERATION MODE (Encrypt/Decrypt Files)
    # =========================================================================
    def setup_operation_tab(self):
        frame = ctk.CTkFrame(self.tab_op)
        frame.pack(pady=20, padx=20, fill="both", expand=True)

        ctk.CTkLabel(frame, text="AUDIO FILE ENCRYPTION", font=ctk.CTkFont(size=22, weight="bold")).pack(pady=10)

        # File
        self.file_path = tk.StringVar()
        ctk.CTkButton(frame, text="📂 Select Audio File", command=self.browse_file).pack(pady=5)
        self.lbl_file = ctk.CTkLabel(frame, text="No file selected", text_color="gray")
        self.lbl_file.pack(pady=5)

        # Settings
        grid = ctk.CTkFrame(frame, fg_color="transparent")
        grid.pack(pady=10)
        ctk.CTkLabel(grid, text="Algorithm:").grid(row=0, column=0, padx=10)
        self.algo_var = ctk.StringVar(value="Hénon Map (2D)")
        algos = ["Hénon Map (2D)", "Logistic Map (1D)", "Lorenz System (3D)", "Chen System (3D)", "ChaCha20", "AES-CTR"]
        ctk.CTkOptionMenu(grid, values=algos, variable=self.algo_var).grid(row=0, column=1, padx=10)

        ctk.CTkLabel(grid, text="Password:").grid(row=0, column=2, padx=10)
        self.entry_pass = ctk.CTkEntry(grid, show="*", width=150)
        self.entry_pass.grid(row=0, column=3, padx=10)

        # Buttons
        btn_box = ctk.CTkFrame(frame, fg_color="transparent")
        btn_box.pack(pady=20)

        # Full Encryption
        ctk.CTkButton(btn_box, text="🔒 FULL LOCK", fg_color="#c0392b", width=120,
                      command=lambda: self.run_full_process("encrypt")).pack(side="left", padx=5)

        # Fast (Selective) Encryption [NEW]
        ctk.CTkButton(btn_box, text="⚡ FAST LOCK (MSB)", fg_color="#d35400", hover_color="#e67e22", width=140,
                      command=lambda: self.run_selective_process("encrypt")).pack(side="left", padx=5)

        # Decrypt
        ctk.CTkButton(btn_box, text="🔓 UNLOCK", fg_color="#27ae60", width=120,
                      command=lambda: self.run_full_process("decrypt")).pack(side="left", padx=5)

        # Log Area
        ctk.CTkLabel(frame, text="Detailed Log Report:", anchor="w").pack(fill="x", padx=20)
        self.op_log = ctk.CTkTextbox(frame, height=250, font=("Consolas", 12))
        self.op_log.pack(pady=5, padx=20, fill="both", expand=True)

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg")])
        if path:
            self.file_path.set(path)
            self.lbl_file.configure(text=os.path.basename(path))

    # --- Full Process Logic ---
    def run_full_process(self, mode):
        threading.Thread(target=self.process_full_thread, args=(mode,), daemon=True).start()

    def process_full_thread(self, mode):
        path = self.file_path.get()
        pwd = self.entry_pass.get()
        algo_name = self.algo_var.get()

        if not path or not pwd:
            self.log_op("Error: Missing file or password.")
            return

        try:
            start_t = time.time()
            self.log_op(f"--- Starting FULL {mode.upper()} ---")

            # Read
            data, sr = sf.read(path, dtype='int16')
            flat = data.flatten()
            byte_data = flat.tobytes()
            n_bytes = len(byte_data)

            # KeyStream
            ks = self.get_keystream(algo_name, pwd, n_bytes)
            if isinstance(ks, bytes): ks = np.frombuffer(ks, dtype=np.uint8)

            # XOR
            processed_bytes = np.bitwise_xor(np.frombuffer(byte_data, dtype=np.uint8), ks)
            final_audio = np.frombuffer(processed_bytes.tobytes(), dtype='int16').reshape(data.shape)

            # Save
            tag = algo_name.split()[0]
            suffix = f"_{tag}_NOISE.wav" if mode == "encrypt" else f"_{tag}_RESTORED.wav"
            out_path = os.path.splitext(path)[0] + suffix
            sf.write(out_path, final_audio, sr)

            self.generate_report(path, mode, algo_name, start_t, "Full Encryption")

        except Exception as e:
            self.log_op(f"Error: {str(e)}")

    # --- Selective (Fast) Process Logic [NEW] ---
    def run_selective_process(self, mode):
        threading.Thread(target=self.process_selective_thread, args=(mode,), daemon=True).start()

    def process_selective_thread(self, mode):
        path = self.file_path.get()
        pwd = self.entry_pass.get()
        algo_name = self.algo_var.get()

        if not path or not pwd:
            self.log_op("Error: Missing file or password.")
            return

        try:
            start_t = time.time()
            self.log_op(f"--- Starting SELECTIVE {mode.upper()} (MSB Only) ---")

            # 1. Read File
            data, sr = sf.read(path, dtype='int16')
            flat_data = data.flatten()

            # 2. Split Critical Parts (MSB vs LSB)
            msb_part = (flat_data >> 8).astype(np.uint8)
            lsb_part = (flat_data & 0xFF).astype(np.uint8)

            # 3. Encrypt ONLY MSB (Half the size = 2x Speed)
            n_bytes = len(msb_part)
            ks = self.get_keystream(algo_name, pwd, n_bytes)
            if isinstance(ks, bytes): ks = np.frombuffer(ks, dtype=np.uint8)

            processed_msb = np.bitwise_xor(msb_part, ks)

            # 4. Reconstruct Audio
            # (Encrypted MSB << 8) | Original LSB
            processed_msb_16 = processed_msb.astype(np.int16) << 8
            final_flat = processed_msb_16 | lsb_part.astype(np.int16)
            final_audio = final_flat.reshape(data.shape)

            # 5. Save
            tag = algo_name.split()[0]
            suffix = f"_{tag}_PARTIAL.wav"  # Mark as Partial
            out_path = os.path.splitext(path)[0] + suffix
            sf.write(out_path, final_audio, sr)

            self.generate_report(path, mode, algo_name, start_t, "Selective (MSB Only)")

        except Exception as e:
            self.log_op(f"Error: {str(e)}")

    def generate_report(self, path, mode, algo, start_time, strategy):
        duration = time.time() - start_time
        file_size = os.path.getsize(path) / (1024 * 1024)
        throughput = file_size / duration if duration > 0 else 0

        report = f"""
--------------------------------------------------
[SUMMARY REPORT]
--------------------------------------------------
File Name     : {os.path.basename(path)}
Strategy      : {strategy}
Algorithm     : {algo}
File Size     : {file_size:.4f} MB
Time Taken    : {duration:.4f} seconds
Throughput    : {throughput:.2f} MB/s
--------------------------------------------------
        """
        self.log_op(report)
        messagebox.showinfo("Success", f"{mode.upper()} Complete!\nMode: {strategy}")

    def log_op(self, text):
        self.op_log.insert("end", text + "\n")
        self.op_log.see("end")

    # =========================================================================
    # TAB 2: RESEARCH/BENCHMARK MODE
    # =========================================================================
    def setup_benchmark_tab(self):
        left = ctk.CTkFrame(self.tab_bench, width=280)
        left.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkLabel(left, text="BENCHMARK LAB", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)

        self.bench_size = ctk.StringVar(value="0.5")
        ctk.CTkLabel(left, text="Data Size (MB):").pack(pady=5)
        ctk.CTkEntry(left, textvariable=self.bench_size).pack(pady=5)

        ctk.CTkButton(left, text="🚀 RUN BENCHMARK", fg_color="#d35400", hover_color="#a04000",
                      height=40, command=self.run_benchmark).pack(pady=20, fill="x", padx=10)

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
            self.bench_log.insert("end", f"Benchmarking on {mb} MB...\n")

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
                self.bench_log.insert("end", f"> Testing {name}...\n")

                start = time.time()
                ks = func(pwd, size_bytes)
                if isinstance(ks, bytes): ks = np.frombuffer(ks, dtype=np.uint8)
                enc = np.bitwise_xor(dummy, ks)
                dur = time.time() - start
                if dur == 0: dur = 0.001

                speed = mb / dur
                entropy = ChaosCryptoCore.calculate_entropy(enc)
                corr = ChaosCryptoCore.calculate_correlation(dummy, enc)

                results.append({"Algo": name, "Speed": speed, "Entropy": entropy, "Corr": corr})

            df = pd.DataFrame(results)
            table_str = df.to_string(formatters={
                'Speed': '{:,.2f}'.format, 'Entropy': '{:,.4f}'.format, 'Corr': '{:,.4f}'.format
            })
            self.bench_log.insert("end", "\n" + "=" * 30 + "\nRESULTS:\n" + "=" * 30 + "\n")
            self.bench_log.insert("end", table_str)

            self.after(0, lambda: self.plot_graphs(df))

        except Exception as e:
            self.bench_log.insert("end", f"\nError: {e}")

    def plot_graphs(self, df):
        for widget in self.right_panel.winfo_children(): widget.destroy()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
        fig.patch.set_facecolor('#2b2b2b')

        ax1.barh(df['Algo'], df['Speed'], color='#3498db')
        ax1.set_title('Encryption Speed (MB/s)', color='white')
        ax1.tick_params(colors='white', labelcolor='white')
        ax1.set_facecolor('#2b2b2b')

        ax2.barh(df['Algo'], df['Entropy'], color='#e74c3c')
        ax2.set_xlim(7.90, 8.01)
        ax2.axvline(8.0, color='lime', linestyle='--')
        ax2.set_title('Entropy (Ideal = 8.0)', color='white')
        ax2.tick_params(colors='white', labelcolor='white')
        ax2.set_facecolor('#2b2b2b')

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
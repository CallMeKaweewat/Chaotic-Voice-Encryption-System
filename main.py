import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import numpy as np
import soundfile as sf
import os
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

# --- การตั้งค่าธีม ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class ChaosAudioEngine:
    """รวมอัลกอริทึม Chaos ทั้งแบบ 1D และ 3D"""

    @staticmethod
    def generate_logistic_map(seed_str, num_values, r=3.99):
        # สร้างค่าเริ่มต้นจาก Password
        x = (hash(seed_str) % 1000000 + 1) / 1000001.0
        sequence = np.zeros(num_values, dtype=np.uint8)
        for i in range(num_values):
            x = r * x * (1 - x)
            sequence[i] = int(x * 255) % 256
        return sequence

    @staticmethod
    def generate_lorenz_system(seed_str, num_values, return_xyz=False):
        x = (hash(seed_str) % 10000) / 1000.0
        y = (hash(seed_str + "y") % 10000) / 1000.0
        z = (hash(seed_str + "z") % 10000) / 1000.0
        sigma, rho, beta, dt = 10.0, 28.0, 8.0 / 3.0, 0.01

        sequence = np.zeros(num_values, dtype=np.uint8)
        xs, ys, zs = [], [], []

        # Warm-up 1000 รอบ
        for _ in range(1000):
            x += sigma * (y - x) * dt
            y += (x * (rho - z) - y) * dt
            z += (x * y - beta * z) * dt

        for i in range(num_values):
            dx = sigma * (y - x) * dt
            dy = (x * (rho - z) - y) * dt
            dz = (x * y - beta * z) * dt
            x, y, z = x + dx, y + dy, z + dz

            sequence[i] = int(abs(x * 1000000)) % 256
            if return_xyz:
                xs.append(x);
                ys.append(y);
                zs.append(z)

        return (sequence, (xs, ys, zs)) if return_xyz else sequence


class AudioCryptoApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Chaotic Voice Encryption System (CVES) Pro v3.0")
        self.geometry("1000x850")

        # --- Variables ---
        self.input_file_path = tk.StringVar()
        self.password = tk.StringVar()
        self.algo_choice = tk.StringVar(value="Lorenz System (3D)")

        # --- UI Layout ---
        self.tabview = ctk.CTkTabview(self, width=950, height=800)
        self.tabview.pack(pady=10, padx=20, fill="both", expand=True)

        self.tab_main = self.tabview.add("Encryption / Decryption")
        self.tab_security = self.tabview.add("Security Analysis")

        self.setup_main_tab()
        self.setup_security_tab()

    def setup_main_tab(self):
        # Header
        ctk.CTkLabel(self.tab_main, text="AUDIO CHAOS VAULT",
                     font=ctk.CTkFont(size=28, weight="bold")).pack(pady=(20, 5))
        ctk.CTkLabel(self.tab_main, text="Secure Stream Cipher based on Non-linear Dynamics",
                     font=ctk.CTkFont(size=14, slant="italic")).pack(pady=(0, 20))

        frame = ctk.CTkFrame(self.tab_main)
        frame.pack(pady=10, padx=40, fill="both", expand=True)

        # File Selection
        ctk.CTkLabel(frame, text="Select .wav File:", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 5))
        ctk.CTkEntry(frame, textvariable=self.input_file_path, width=500).pack(pady=5)
        ctk.CTkButton(frame, text="Browse File", command=self.browse_file).pack(pady=5)

        # Algorithm Selection
        ctk.CTkLabel(frame, text="Select Algorithm:", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 5))
        ctk.CTkOptionMenu(frame, values=["Logistic Map (1D)", "Lorenz System (3D)"],
                          variable=self.algo_choice).pack(pady=5)

        # Secret Key
        ctk.CTkLabel(frame, text="Enter Secret Key:", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 5))
        ctk.CTkEntry(frame, textvariable=self.password, show="*", width=300).pack(pady=5)

        # Action Buttons
        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(pady=30)
        ctk.CTkButton(btn_frame, text="ENCRYPT", fg_color="#c0392b", hover_color="#a93226",
                      command=lambda: self.process_audio(True)).grid(row=0, column=0, padx=20)
        ctk.CTkButton(btn_frame, text="DECRYPT", fg_color="#27ae60", hover_color="#1e8449",
                      command=lambda: self.process_audio(False)).grid(row=0, column=1, padx=20)

        # Visualization Toggle
        ctk.CTkButton(frame, text="Show Chaotic Attractor Visual", fg_color="#34495e",
                      command=self.show_visualizations).pack(pady=10)

        self.status_label = ctk.CTkLabel(self.tab_main, text="Status: Ready", font=ctk.CTkFont(size=14))
        self.status_label.pack(pady=10)

    def setup_security_tab(self):
        self.test_pass1 = ctk.StringVar(value="Key_Alpha_1")
        self.test_pass2 = ctk.StringVar(value="Key_Alpha_2")

        ctk.CTkLabel(self.tab_security, text="CHAOS SENSITIVITY TEST (BUTTERFLY EFFECT)",
                     font=ctk.CTkFont(size=22, weight="bold")).pack(pady=20)

        test_frame = ctk.CTkFrame(self.tab_security)
        test_frame.pack(pady=10, padx=20, fill="x")

        ctk.CTkLabel(test_frame, text="Key A:").grid(row=0, column=0, padx=5, pady=10)
        ctk.CTkEntry(test_frame, textvariable=self.test_pass1).grid(row=0, column=1, padx=5)
        ctk.CTkLabel(test_frame, text="Key B:").grid(row=0, column=2, padx=5)
        ctk.CTkEntry(test_frame, textvariable=self.test_pass2).grid(row=0, column=3, padx=5)

        ctk.CTkButton(test_frame, text="Run Security Analysis",
                      command=self.run_security_test, fg_color="#E67E22").grid(row=0, column=4, padx=10)

        self.analysis_label = ctk.CTkLabel(self.tab_security, text="Difference Score: -- %",
                                           font=ctk.CTkFont(size=16, weight="bold"))
        self.analysis_label.pack(pady=5)

        self.viz_frame = ctk.CTkFrame(self.tab_security, fg_color="#2b2b2b")
        self.viz_frame.pack(pady=10, padx=20, fill="both", expand=True)
        self.sec_canvas = None

    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if filename:
            self.input_file_path.set(filename)
            self.status_label.configure(text=f"Loaded: {os.path.basename(filename)}", text_color="white")

    def process_audio(self, is_encrypt):
        path, pwd = self.input_file_path.get(), self.password.get()
        if not path or not pwd:
            messagebox.showwarning("Warning", "Please select a file and enter a secret key.")
            return

        try:
            start = time.time()
            data, sr = sf.read(path, dtype='int16')
            original_shape = data.shape
            flat_data = data.flatten()
            audio_bytes = flat_data.tobytes()

            # เลือก Algorithm ตามที่ User เลือก
            if "Lorenz" in self.algo_choice.get():
                ks = ChaosAudioEngine.generate_lorenz_system(pwd, len(audio_bytes))
            else:
                ks = ChaosAudioEngine.generate_logistic_map(pwd, len(audio_bytes))

            # XOR Encryption
            processed_bytes = np.bitwise_xor(np.frombuffer(audio_bytes, dtype=np.uint8), ks)

            # แปลงกลับเป็น Audio Format
            final_data = np.frombuffer(processed_bytes.tobytes(), dtype='int16').reshape(original_shape)

            suffix = "_encrypted.wav" if is_encrypt else "_decrypted.wav"
            out = os.path.splitext(path)[0] + suffix
            sf.write(out, final_data, sr)

            self.status_label.configure(text=f"Success! Done in {time.time() - start:.4f}s", text_color="#2ecc71")
            messagebox.showinfo("Success", f"File saved to:\n{out}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def show_visualizations(self):
        pwd = self.password.get()
        if not pwd:
            messagebox.showwarning("Warning", "Please enter a password to see the Chaos Map.")
            return

        if "Lorenz" in self.algo_choice.get():
            _, (xs, ys, zs) = ChaosAudioEngine.generate_lorenz_system(pwd, 20000, return_xyz=True)
            fig = plt.figure("Lorenz Attractor (3D Chaos)", figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(xs, ys, zs, lw=0.5, color='magenta')
            ax.set_title(f"Lorenz Attractor Path for Key: {pwd}")
            plt.show()
        else:
            # สำหรับ Logistic Map แสดงผลแบบ Time Series
            ks = ChaosAudioEngine.generate_logistic_map(pwd, 1000)
            plt.figure("Logistic Map (1D Chaos)", figsize=(8, 4))
            plt.plot(ks[:200], 'g-', lw=1)
            plt.title(f"Logistic Map Sequence (First 200 pts) for Key: {pwd}")
            plt.show()

    def run_security_test(self):
        p1, p2 = self.test_pass1.get(), self.test_pass2.get()
        length = 1000

        # ใช้ Lorenz เป็นมาตรฐานในการทดสอบ Sensitivity
        k1 = ChaosAudioEngine.generate_lorenz_system(p1, length)
        k2 = ChaosAudioEngine.generate_lorenz_system(p2, length)

        diff = (np.sum(k1 != k2) / length) * 100
        self.analysis_label.configure(text=f"Butterfly Effect Score: {diff:.2f}% Difference", text_color="#E67E22")

        if self.sec_canvas:
            self.sec_canvas.get_tk_widget().destroy()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))
        plt.subplots_adjust(hspace=0.6)

        ax1.plot(k1[:300], color='#3498db')
        ax1.set_title(f"Sequence A (Key: '{p1}')")
        ax1.set_ylabel("Value")

        ax2.plot(k2[:300], color='#e74c3c')
        ax2.set_title(f"Sequence B (Key: '{p2}')")
        ax2.set_ylabel("Value")

        self.sec_canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        self.sec_canvas.draw()
        self.sec_canvas.get_tk_widget().pack(fill="both", expand=True)


if __name__ == "__main__":
    app = AudioCryptoApp()
    app.mainloop()
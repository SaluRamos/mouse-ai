import tkinter as tk
import random
import csv
import time
import threading
import ctypes
import logging

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

def get_mouse_pos():
    pt = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

capture_frequency = 50
capture_sleep = 1/capture_frequency

class App:

    def __init__(self, root):
        self.recording = False
        self.root = root
        self.root.title("Obter dados")
        self.root.bind("<Configure>", self.on_resize)
        self.width = 200
        self.height = 200
        self.btn_w = 20
        self.btn_h = 5
        self.root.geometry(f"{self.width}x{self.height}")
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg="#f0f0f0")
        self.canvas.pack(fill="both", expand=True)
        #random button
        self.total_clicks = 0
        self.random_btn = tk.Button(
            self.root, 
            text="Clique aqui", 
            padx=self.btn_w,
            pady=self.btn_h,
            command=self.define_random_target,
            bg="green",
            compound="c"
        )
        self.place_btn_at_center()
        #timer
        self.timer_lbl = tk.Label(
            self.root,
            text="0.0 s",
            font=("Arial", 14),
            bg="#f0f0f0"
        )
        self.timer_lbl.place(relx=0.5, y=10, anchor="n")
        #data utils
        self.clicked = False
        self.clicked_lock = threading.Lock()

    def get_data(self):
        global capture_sleep
        #a captação deve ser feita em intervalos iguais
        try:
            with open(f"data-{time.time()}.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "pos_x","pos_y",
                    "target_x","target_y",
                    "is_inside_btn",
                    "offset_x","offset_y",
                    "click",
                    "mov_x","mov_y"
                ])
                last_m_pos_x, last_m_pos_y = get_mouse_pos()
                while self.recording:
                    m_pos_x, m_pos_y = get_mouse_pos()

                    root_x = self.root.winfo_rootx()
                    root_y = self.root.winfo_rooty()

                    btn_global_x = root_x + self.btn_x
                    btn_global_y = root_y + self.btn_y
                    is_mouse_inside_btn = (btn_global_x <= m_pos_x <= btn_global_x + self.bw and btn_global_y <= m_pos_y <= btn_global_y + self.bh)

                    target_middle_x = root_x + self.target_x
                    target_middle_y = root_y + self.target_y
                    offset_x = (target_middle_x - m_pos_x)/self.width
                    offset_y = (target_middle_y - m_pos_y)/self.height
                    #A - B = BA
                    mov_x = (m_pos_x - last_m_pos_x)/self.width
                    mov_y = (m_pos_y - last_m_pos_y)/self.height
                    #escrever normalizado
                    writer.writerow([
                        m_pos_x, m_pos_y,
                        target_middle_x, target_middle_y, 
                        int(is_mouse_inside_btn),
                        offset_x, offset_y,
                        int(self.clicked),
                        mov_x, mov_y
                    ])
                    with self.clicked_lock:
                        if self.clicked:
                            self.clicked = False
                    last_m_pos_x = m_pos_x
                    last_m_pos_y = m_pos_y
                    time.sleep(capture_sleep)
        except Exception as e:
            logging.error("Falha na thread", exc_info=True)

    def on_resize(self, event):
        if event.widget is self.root:
            self.width = event.width
            self.height = event.height
            if not self.recording:
                self.place_btn_at_center()

    def update_timer(self):
        if self.recording:
            elapsed = time.time() - self.start_time
            self.timer_lbl.config(text=f"{elapsed:.1f} s")
            self.root.after(100, self.update_timer)

    def place_btn_at_center(self):
        self.bw = self.random_btn.winfo_width()
        self.bh = self.random_btn.winfo_height()
        self.random_btn.place(x=(self.width/2) - self.btn_w*2, y=(self.height/2) - self.btn_h)

    def define_random_target(self):
        with self.clicked_lock:
            self.clicked = True
        self.total_clicks = self.total_clicks + 1
        if not self.recording:
            self.recording = True
            self.start_time = time.time()
            self.update_timer()
            threading.Thread(target=self.get_data, daemon=True).start()
        self.root.update_idletasks()
        self.bw = self.random_btn.winfo_width()
        self.bh = self.random_btn.winfo_height()
        self.btn_x = random.randint(0, self.width - self.bw)
        self.btn_y = random.randint(0, self.height - self.bh)
        self.target_x = self.btn_x + self.bw/2
        self.target_y = self.btn_y + self.bh/2
        self.random_btn.place(x=self.btn_x, y=self.btn_y)
        print(f"Novo alvo: x={self.btn_x}, y={self.btn_y}, clicks={self.total_clicks}")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
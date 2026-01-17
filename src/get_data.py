#libs
import tkinter as tk
import keyboard
#native libs
import random
import csv
import time
import threading
import ctypes
import logging

SHORTCUT_QUIT = 'ctrl+o'
RANDOMIZE_BTN_SIZE = True
capture_sleep = 1 #millisecond

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

def get_mouse_pos() -> tuple[int, int]:
    pt = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

class App:

    def __init__(self, root):
        self.recording = False
        self.root = root
        self.root.title("Obter dados")
        self.root.bind("<Configure>", self.on_resize)
        self.width = 800
        self.height = 800
        self.root.geometry(f"{self.width}x{self.height}")
        #timer
        self.timer_lbl = tk.Label(
            self.root,
            text="0.0 s",
            font=("Arial", 14),
            bg="#f0f0f0"
        )
        self.timer_lbl.place(relx=0.5, y=10, anchor="n")
        #random button
        self.total_clicks = 0
        self.bw = 40
        self.bh = 20
        self.random_btn = tk.Button(
            self.root, 
            text="Start",
            font=("Arial", 10),
            command=self.define_random_target,
            bg="green"
        )
        self.random_btn.place(x=0, y=0, width=self.bw, height=self.bh)
        #data utils
        self.clicked = False
        self.clicked_lock = threading.Lock()

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
        self.random_btn.place(x=(self.width - self.bw)/2 , y=(self.height - self.bh)/2, width=self.bw, height=self.bh)

    def define_random_target(self):
        with self.clicked_lock:
            self.clicked = True
        self.total_clicks = self.total_clicks + 1
        if not self.recording:
            self.recording = True
            self.random_btn.config(text="X", bg="red")
            self.start_time = time.time()
            self.update_timer()
            threading.Thread(target=self.get_data, daemon=True).start()
        self.root.update_idletasks()
        if RANDOMIZE_BTN_SIZE:
            self.bw = random.randint(10, int(self.width/5))
            self.bh = random.randint(10, int(self.height/5))
        self.bx = random.randint(0, self.width - self.bw)
        self.by = random.randint(0, self.height - self.bh)
        self.random_btn.place(x=self.bx, y=self.by, width=self.bw, height=self.bh)
        print(f"Novo alvo: x={self.bx}, y={self.by} w={self.bw}, h={self.bh}, clicks={self.total_clicks}")

    def get_data(self):
        global capture_sleep
        #a captação deve ser feita em intervalos iguais
        try:
            with open(f"data/data-{time.time()}.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "global_pos_x","global_pos_y",
                    "root_x","root_y",
                    "window_width","window_height",
                    "is_inside_btn",
                    "offset_x","offset_y",
                    "click",
                    "mov_x","mov_y",
                    "btn_w","btn_h"
                ])
                last_m_pos_x, last_m_pos_y = get_mouse_pos()

                captures = 0
                last_capture = time.time()

                while self.recording:
                    captures += 1
                    now = time.time()
                    if now - last_capture >= 1:
                        print(f"CAPTURES LAST SEC: {captures}")
                        captures = 0
                        last_capture = now

                    m_pos_x, m_pos_y = get_mouse_pos()

                    root_x = self.root.winfo_rootx()
                    root_y = self.root.winfo_rooty()

                    btn_global_x = root_x + self.bx
                    btn_global_y = root_y + self.by
                    is_mouse_inside_btn = (btn_global_x <= m_pos_x <= btn_global_x + self.bw and btn_global_y <= m_pos_y <= btn_global_y + self.bh)

                    target_x = self.bx + self.bw/2
                    target_y = self.by + self.bh/2

                    target_middle_x = root_x + target_x
                    target_middle_y = root_y + target_y
                    offset_x = (target_middle_x - m_pos_x)/self.width
                    offset_y = (target_middle_y - m_pos_y)/self.height

                    norm_bw = self.bw/self.width
                    norm_bh = self.bh/self.height
                    if keyboard.is_pressed(SHORTCUT_QUIT):
                        print("finalizou captura")
                        self.recording = False
                        break
                    #A - B = BA
                    mov_x = (m_pos_x - last_m_pos_x)/self.width
                    mov_y = (m_pos_y - last_m_pos_y)/self.height
                    #escrever normalizado
                    writer.writerow([
                        m_pos_x, m_pos_y,
                        root_x,root_y,
                        self.width, self.height,
                        int(is_mouse_inside_btn),
                        offset_x, offset_y,
                        int(self.clicked),
                        mov_x, mov_y,
                        norm_bw, norm_bh
                    ])
                    with self.clicked_lock:
                        if self.clicked:
                            self.clicked = False
                    last_m_pos_x = m_pos_x
                    last_m_pos_y = m_pos_y
                    ctypes.windll.kernel32.Sleep(capture_sleep)
        except Exception as e:
            logging.error("Falha na thread", exc_info=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
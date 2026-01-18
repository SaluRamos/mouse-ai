#modules
from utils import get_mouse_pos
#libs
import tkinter as tk
import keyboard
#native libs
import random
import csv
import time
import threading
import logging
import os

SHORTCUT_QUIT = 'ctrl+o'
RANDOMIZE_BTN_SIZE = True

#125Hz é taxa de atualização padrão de mouses
#mas os mecanismo de detecção de bots usam o 'mousemove' do javascript
#que costuma ser disparado usando a informação do monitor, na maioria dos casos 60Hz
TARGET_HZ = 60
PERIOD = 1.0 / TARGET_HZ

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
        #next random button
        self.next_random_btn = tk.Button(
            self.root, 
            text="",
            font=("Arial", 10),
            command=self.define_random_target,
            bg="#FFFFFF",
            state="disabled"
        )
        self.define_next_random_target()
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

    def define_next_random_target(self):
        self.root.update_idletasks()
        if RANDOMIZE_BTN_SIZE:
            self.next_bw = random.randint(10, int(self.width/5))
            self.next_bh = random.randint(10, int(self.height/5))
        self.next_bx = random.randint(0, self.width - self.next_bw)
        self.next_by = random.randint(0, self.height - self.next_bh)
        self.next_random_btn.place(x=self.next_bx, y=self.next_by, width=self.next_bw, height=self.next_bh)

    def define_random_target(self):
        with self.clicked_lock:
            self.clicked = True
        self.total_clicks += 1
        if not self.recording:
            self.recording = True
            self.random_btn.config(text="X", bg="red")
            self.start_time = time.time()
            self.update_timer()
            threading.Thread(target=self.get_data, daemon=True).start()
        self.root.update_idletasks()
        if RANDOMIZE_BTN_SIZE:
            self.bw = self.next_bw
            self.bh = self.next_bh
        self.bx = self.next_bx
        self.by = self.next_by
        self.define_next_random_target()
        self.random_btn.place(x=self.bx, y=self.by, width=self.bw, height=self.bh)
        print(f"Novo alvo: x={self.bx}, y={self.by} w={self.bw}, h={self.bh}, clicks={self.total_clicks}")

    def get_data(self):
        global capture_sleep
        #a captação deve ser feita em intervalos iguais
        try:
            id = str(time.time()).replace(".", "")
            name = f"data/data-{id}.csv"
            if not os.path.exists("data"):
                name = "../" + name
            with open(name, "w+", newline="") as f:
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
                next_time = time.perf_counter()

                while self.recording:

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

                    next_time += PERIOD
                    sleep_time = next_time - time.perf_counter()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        except Exception as e:
            logging.error("Falha na thread", exc_info=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
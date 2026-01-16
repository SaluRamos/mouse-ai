import tensorflow as tf
import tkinter as tk
import random
import csv
import time
import ctypes
import threading
import keyboard
from collections import deque
import numpy as np
import math
from enums import ModelType

MODEL_TYPE = ModelType.RNN
SHORTCUT_QUIT = 'ctrl+o'
RANDOMIZE_BTN_SIZE = True
EXECUTE_AI = True

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

def get_mouse_pos() -> tuple[int, int]:
    pt = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

SendInput = ctypes.windll.user32.mouse_event
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

def mov_mouse(mov_x:int, mov_y:int, click:bool) -> None:
    SendInput(MOUSEEVENTF_MOVE, mov_x, mov_y, 0, 0)
    if click:
        SendInput(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        SendInput(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

class App:
    
    def __init__(self, root):
        self.started = False
        self.root = root
        self.root.title("Obter dados")
        self.root.bind("<Configure>", self.on_resize)
        self.width = 800
        self.height = 800
        self.root.geometry(f"{self.width}x{self.height}")
        self.bw = 40
        self.bh = 20
        #random button
        self.total_clicks = 0
        self.random_btn = tk.Button(
            self.root, 
            text="Start",
            font=("Arial", 10),
            command=self.define_random_target,
            bg="green"
        )
        self.random_btn.place(x=0, y=0, width=self.bw, height=self.bh)

    def ai_thread(self):
        if MODEL_TYPE == ModelType.MLP:
            MODEL_PATH = "models/mouse_mlp.keras"
        if MODEL_TYPE == ModelType.RNN:
            MODEL_PATH = "models/mouse_rnn.keras"
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        btn_size_m = 1
        inv_btn_size_m = (1 - btn_size_m)
        # Criamos um buffer que mantém apenas os últimos 10 registros para a RNN
        time_steps = 20
        buffer = deque(maxlen=time_steps)
        for _ in range(time_steps):
            buffer.append([0.0, 0.0, 0.0, 0.0, 0.0])
        # loop de execução
        while True:
            if keyboard.is_pressed(SHORTCUT_QUIT):
                print("IA finalizada")
                self.started = False
                break
            m_pos_x, m_pos_y = get_mouse_pos()

            root_x = self.root.winfo_rootx()
            root_y = self.root.winfo_rooty()

            btn_global_x = root_x + self.bx
            btn_global_y = root_y + self.by
            is_mouse_inside_btn = (btn_global_x + self.bw*inv_btn_size_m <= m_pos_x <= btn_global_x + self.bw*btn_size_m and 
                                   btn_global_y + self.bh*inv_btn_size_m <= m_pos_y <= btn_global_y + self.bh*btn_size_m)
            
            target_x = self.bx + self.bw/2
            target_y = self.by + self.bh/2

            # inputs (normalizados)
            target_middle_x = root_x + target_x
            target_middle_y = root_y + target_y
            offset_x = (target_middle_x - m_pos_x)/self.width
            offset_y = (target_middle_y - m_pos_y)/self.height

            norm_bw = (self.bw/self.width)*btn_size_m
            norm_bh = (self.bh/self.height)*btn_size_m
            # inferência
            inp = None
            if MODEL_TYPE == ModelType.MLP:
                inp = tf.convert_to_tensor([[offset_x, offset_y, is_mouse_inside_btn, norm_bw, norm_bh]], dtype=tf.float32)
            if MODEL_TYPE == ModelType.RNN:
                buffer.append([offset_x, offset_y, is_mouse_inside_btn, norm_bw, norm_bh])
                inp = np.array([list(buffer)], dtype=np.float32)
            mov_x_n, mov_y_n, click_p = model.predict(inp, verbose=0)
            # desnormaliza movimento
            mov_x = math.ceil(mov_x_n[0][0] * self.width)
            mov_y = math.ceil(mov_y_n[0][0] * self.height)
            # threshold de clique
            click = False
            if MODEL_TYPE == ModelType.MLP:
                click = click_p[0][0] > 0.01 
            if MODEL_TYPE == ModelType.RNN:
                click = click_p[0][0] > 0.2
            #efetuar ação da IA
            if EXECUTE_AI:
                mov_mouse(mov_x, mov_y, click and is_mouse_inside_btn)
            print(f"mov_x={round(mov_x, 3)}, mov_y={round(mov_y, 3)}, inside_btn={is_mouse_inside_btn}, click={round(click_p[0][0], 3)}")

    def on_resize(self, event):
        if event.widget is self.root:
            self.width = event.width
            self.height = event.height
            if not self.started:
                self.place_btn_at_center()

    def place_btn_at_center(self):
        self.random_btn.place(x=(self.width - self.bw)/2 , y=(self.height - self.bh)/2, width=self.bw, height=self.bh)

    def define_random_target(self):
        self.total_clicks = self.total_clicks + 1
        if not self.started:
            self.started = True
            threading.Thread(target=self.ai_thread, daemon=True).start()
            self.random_btn.config(text="X", bg="red")
        self.root.update_idletasks()
        if RANDOMIZE_BTN_SIZE:
            self.bw = random.randint(10, self.width/5)
            self.bh = random.randint(10, self.height/5)
        self.bx = random.randint(0, self.width - self.bw)
        self.by = random.randint(0, self.height - self.bh)
        self.random_btn.place(x=self.bx, y=self.by, width=self.bw, height=self.bh)
        print(f"Novo alvo: x={self.bx}, y={self.by} w={self.bw}, h={self.bh}, clicks={self.total_clicks}")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
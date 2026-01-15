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
from enum import Enum
import math

class ModelType(Enum):
    MLP = 0
    RNN = 1

MODEL_TYPE = ModelType.MLP
SHORTCUT_QUIT = 'ctrl+o'
EXECUTE_AI = True
MAX_AI_MOV_MAGNITUDE = 30

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

def vector_magnitude(x:int, y:int) -> float:
    return (x**2 + y**2)**0.5

def adjust_vector_magnitude(x:int, y:int, desired_magnitude:float) -> tuple[int, int]:
    actual_magnitude = vector_magnitude(x, y)
    if actual_magnitude == 0:
        return 0, 0
    new_x = (x / actual_magnitude) * desired_magnitude
    new_y = (y / actual_magnitude) * desired_magnitude
    return math.ceil(new_x), math.ceil(new_y)

class App:
    
    def __init__(self, root):
        self.started = False
        self.root = root
        self.root.title("Obter dados")
        self.root.bind("<Configure>", self.on_resize)
        self.width = 800
        self.height = 800
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

    def ai_thread(self):
        if MODEL_TYPE == ModelType.MLP:
            MODEL_PATH = "models/mouse_mlp.keras"
        if MODEL_TYPE == ModelType.RNN:
            MODEL_PATH = "models/mouse_rnn.keras"
        model = tf.keras.models.load_model(MODEL_PATH)
        # Criamos um buffer que mantém apenas os últimos 10 registros para a RNN
        time_steps = 10
        buffer = deque(maxlen=time_steps)
        for _ in range(time_steps):
            buffer.append([0.0, 0.0, 0.0])
        # loop de execução
        while True:
            if keyboard.is_pressed(SHORTCUT_QUIT):
                print("IA finalizada")
                self.started = False
                break
            m_pos_x, m_pos_y = get_mouse_pos()

            root_x = self.root.winfo_rootx()
            root_y = self.root.winfo_rooty()

            btn_global_x = root_x + self.btn_x
            btn_global_y = root_y + self.btn_y
            is_mouse_inside_btn = (btn_global_x <= m_pos_x <= btn_global_x + self.bw and btn_global_y <= m_pos_y <= btn_global_y + self.bh)
            
            target_x = self.btn_x + self.bw/2
            target_y = self.btn_y + self.bh/2

            # inputs (normalizados)
            target_middle_x = root_x + target_x
            target_middle_y = root_y + target_y
            offset_x = (target_middle_x - m_pos_x)/self.width
            offset_y = (target_middle_y - m_pos_y)/self.height
            # inferência
            inp = None
            if MODEL_TYPE == ModelType.MLP:
                inp = tf.convert_to_tensor([[offset_x, offset_y, is_mouse_inside_btn]], dtype=tf.float32)
            if MODEL_TYPE == ModelType.RNN:
                buffer.append([offset_x, offset_y, is_mouse_inside_btn])
                inp = np.array([list(buffer)], dtype=np.float32)
            mov_x_n, mov_y_n, click_p = model.predict(inp, verbose=0)
            # desnormaliza movimento
            mov_x = math.ceil(mov_x_n[0][0] * self.width)
            mov_y = math.ceil(mov_y_n[0][0] * self.height)
            mag = vector_magnitude(mov_x, mov_y)
            if mag > MAX_AI_MOV_MAGNITUDE:
                mov_x, mov_y = adjust_vector_magnitude(mov_x, mov_y, MAX_AI_MOV_MAGNITUDE)
            # threshold de clique
            click = False
            if MODEL_TYPE == ModelType.MLP:
                click = click_p[0][0] < 0.01 
            if MODEL_TYPE == ModelType.RNN:
                click = click_p[0][0] > 0.2
            #efetuar ação da IA
            if EXECUTE_AI:
                mov_mouse(mov_x, mov_y, click and is_mouse_inside_btn)
            print(f"{round(mov_x, 3)}, {round(mov_y, 3)}, {click}, {round(click_p[0][0], 3)}")

    def on_resize(self, event):
        if event.widget is self.root:
            self.width = event.width
            self.height = event.height
            self.place_btn_at_center()

    def place_btn_at_center(self):
        self.bw = self.random_btn.winfo_width()
        self.bh = self.random_btn.winfo_height()
        self.random_btn.place(x=(self.width/2) - self.btn_w*2, y=(self.height/2) - self.btn_h)

    def define_random_target(self):
        self.total_clicks = self.total_clicks + 1
        if not self.started:
            self.started = True
            threading.Thread(target=self.ai_thread, daemon=True).start()
        self.root.update_idletasks()
        self.bw = self.random_btn.winfo_width()
        self.bh = self.random_btn.winfo_height()
        self.btn_x = random.randint(0, self.width - self.bw)
        self.btn_y = random.randint(0, self.height - self.bh)
        self.random_btn.place(x=self.btn_x, y=self.btn_y)
        print(f"Novo alvo: x={self.btn_x}, y={self.btn_y}, clicks={self.total_clicks}")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
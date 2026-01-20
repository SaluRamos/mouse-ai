#modules
from utils import get_mouse_pos, mov_mouse, click_mouse
from enums import ModelType
from ticks_inside_btn_until_click import random_ticks_inside_btn
#libs
import tensorflow as tf
import keyboard
import numpy as np
#native libs
import tkinter as tk
import random
import threading
from collections import deque
import math
import os
import time

MODEL_TYPE = ModelType.MLP
SHORTCUT_QUIT = 'ctrl+o'
RANDOMIZE_BTN_SIZE = True
EXECUTE_AI = True

TARGET_HZ = 60
PERIOD = 1.0 / TARGET_HZ

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

    # # Exemplo de uso futuro
    # generator = tf.keras.models.load_model("models/mouse_gan_generator.keras")
    # # Quero mover para um alvo que está em x=0.5, y=-0.2 de distância
    # offset = np.array([[0.5, -0.2, 0.05, 0.02]]) # offset_x, offset_y, btn_w, btn_h
    # noise = np.random.normal(0, 1, (1, 100)) # 100 é o LATENT_DIM
    # path = generator.predict([noise, offset]) 
    # # path shape será (1, 100, 2) -> 100 passos de (dx, dy)

    def ai_thread(self):
        if MODEL_TYPE == ModelType.MLP:
            MODEL_PATH = "models/mouse_mlp.keras"
        if MODEL_TYPE == ModelType.RNN:
            MODEL_PATH = "models/mouse_rnn.keras"
        if not os.path.exists(MODEL_PATH):
            MODEL_PATH = "../" + MODEL_PATH
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)

        btn_size_m = 1
        inv_btn_size_m = (1 - btn_size_m)

        time_steps = 10
        buffer = deque(maxlen=time_steps)
        if MODEL_TYPE == ModelType.RNN:
            for _ in range(time_steps):
                buffer.append([0.0, 0.0, 0.0, 0.0, 0.0])

        #pre calculated
        norm_bw = (self.bw/self.width)*btn_size_m
        norm_bh = (self.bh/self.height)*btn_size_m
        inp = None
        ticks_inside_btn = 0
        next_ticks_to_click = random_ticks_inside_btn()

        next_time = time.perf_counter()

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
            is_mouse_inside_btn = (
                                btn_global_x + self.bw*inv_btn_size_m <= m_pos_x <= btn_global_x + self.bw*btn_size_m and 
                                btn_global_y + self.bh*inv_btn_size_m <= m_pos_y <= btn_global_y + self.bh*btn_size_m)
            target_x = self.bx + self.bw/2
            target_y = self.by + self.bh/2
            target_middle_x = root_x + target_x
            target_middle_y = root_y + target_y
            offset_x = (target_middle_x - m_pos_x)/self.width
            offset_y = (target_middle_y - m_pos_y)/self.height

            if is_mouse_inside_btn:
                ticks_inside_btn += 1

            if ticks_inside_btn > 0 and not is_mouse_inside_btn:
                ticks_inside_btn = 0

            # inference
            if MODEL_TYPE == ModelType.MLP:
                inp = tf.convert_to_tensor([[offset_x, offset_y, is_mouse_inside_btn, norm_bw, norm_bh]], dtype=tf.float32)
            if MODEL_TYPE == ModelType.RNN:
                buffer.append([offset_x, offset_y, is_mouse_inside_btn, norm_bw, norm_bh])
                inp = np.array([list(buffer)], dtype=np.float32)
            
            mov_x_n, mov_y_n = model.predict(inp, verbose=0)
            mov_x = math.ceil(mov_x_n[0][0] * self.width)
            mov_y = math.ceil(mov_y_n[0][0] * self.height)

            if EXECUTE_AI:
                mov_mouse(mov_x, mov_y)
                if ticks_inside_btn >= next_ticks_to_click and is_mouse_inside_btn:
                    click_mouse()
                    next_ticks_to_click = random_ticks_inside_btn()
                    ticks_inside_btn = 0
            print(f"mov_x={mov_x:<8.3f}, mov_y={mov_y:<8.3f}, inside_btn={is_mouse_inside_btn:<2.0f}, ticks_inside={ticks_inside_btn:<2.0f}, next_click={next_ticks_to_click:<2.0f}")

            next_time += PERIOD
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

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
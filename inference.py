import tensorflow as tf
import tkinter as tk
import random
import csv
import time
import ctypes
import threading
import keyboard

SHORTCUT_QUIT = 'ctrl+o'
capture_frequency = 50
capture_sleep = 1/capture_frequency

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

    def ai_thread(self):
        global capture_sleep
        #Como interpretar a saída da rede neural
        #desnormalize a saída (mov_x e mov_y)
        #threshold para a saída click
        model = tf.keras.models.load_model("mouse_rnn.keras")
        while True:
            if keyboard.is_pressed(SHORTCUT_QUIT):
                print("IA finalizada")
                self.started = False
                break
            m_pos_x, m_pos_y = get_mouse_pos()

            root_x = self.root.winfo_rootx()
            root_y = self.root.winfo_rooty()

            target_middle_x = root_x + self.target_x + self.bw/2
            target_middle_y = root_y + self.target_y + self.bh/2

            btn_global_x = root_x + self.btn_x
            btn_global_y = root_y + self.btn_y

            is_mouse_inside_btn = (btn_global_x <= m_pos_x <= btn_global_x + self.bw and btn_global_y <= m_pos_y <= btn_global_y + self.bh)
            # centro do botão
            # inputs (normalizados)
            offset_x = (target_middle_x - m_pos_x)/self.width
            offset_y = (target_middle_y - m_pos_y)/self.height
            inp = tf.convert_to_tensor([[offset_x, offset_y, is_mouse_inside_btn]], dtype=tf.float32)
            # inferência
            mov_x_n, mov_y_n, click_p = model.predict(inp, verbose=0)
            # desnormaliza movimento
            mov_x = int(mov_x_n[0][0] * self.width)
            mov_y = int(mov_y_n[0][0] * self.height)
            click = click_p[0][0] > 0.45
            # threshold de clique
            print(f"{mov_x}, {mov_y}, {click_p[0][0]}")
            # print(f"{offset_x}, {offset_y}")
            mov_mouse(mov_x, mov_y, False)
            time.sleep(capture_sleep)

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
        self.target_x = self.btn_x + self.bw/2
        self.target_y = self.btn_y + self.bh/2
        self.random_btn.place(x=self.btn_x, y=self.btn_y)
        print(f"Novo alvo: x={self.btn_x}, y={self.btn_y}, clicks={self.total_clicks}")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
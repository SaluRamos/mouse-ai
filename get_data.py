import tkinter as tk
import random
import csv
import time

class App:
    def __init__(self, root):
        self.recording = False
        self.root = root
        self.root.title("Obter dados")

        self.root.bind("<Configure>", self.on_resize)
        self.width = 200
        self.height = 200
        self.btn_w = 40
        self.btn_h = 10

        self.root.geometry(f"{self.width}x{self.height}")

        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg="#f0f0f0")
        self.canvas.pack(fill="both", expand=True)

        self.random_btn = tk.Button(
            self.root, 
            text="Clique aqui", 
            padx=self.btn_w,
            pady=self.btn_h,
            command=self.define_random_target,
            bg="green",
            compound="c"
        )
        self.random_btn.place(x=(self.width/2) - self.btn_w, y=(self.btn_h/2) - self.btn_h)

    def on_resize(self, event):
        if event.widget is self.root:
            self.width = event.width
            self.height = event.height


    def define_random_target(self):
        self.root.update_idletasks()
        bw = self.random_btn.winfo_width()
        bh = self.random_btn.winfo_height()
        if not self.recording:
            self.recording = True
        self.target_x = random.randint(0, self.width - bw)
        self.target_y = random.randint(0, self.height - bh)
        self.random_btn.place(x=self.target_x, y=self.target_y)
        print(f"Novo alvo: x={self.target_x}, y={self.target_y}")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
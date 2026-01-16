import tensorflow as tf
from tensorflow.keras import layers, models
import glob
import csv
import numpy as np


def load_csv_data(path="data/data-*.csv"):
    X = []
    y = []
    files = glob.glob(path)
    removed_lines = 0
    for file in files:
        print(f"[INFO] lendo: {file}")
        with open(file, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if float(row["mov_x"]) == 0 and float(row["mov_y"]) == 0:
                    removed_lines += 1
                    continue
                X.append([
                    float(row["offset_x"]),
                    float(row["offset_y"]),
                    float(row["is_inside_btn"]),
                    float(row["btn_w"]),
                    float(row["btn_h"])
                ])
                y.append([
                    float(row["mov_x"]),
                    float(row["mov_y"]),
                    float(row["click"])
                ])
                if i == 0:
                    print("[DEBUG] primeira linha:", X[-1], y[-1])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    print(f"REMOVED LINES WITH MOV 0 = {removed_lines}")
    print(f"[INFO] amostras carregadas: {X.shape[0]}")
    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")
    return X, y

def create_sequences(X, y, time_steps:int):
    Xs, ys_mov_x, ys_mov_y, ys_click = [], [], [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        # O alvo é sempre o resultado do ÚLTIMO frame da sequência
        ys_mov_x.append(y[i + time_steps][0])
        ys_mov_y.append(y[i + time_steps][1])
        ys_click.append(y[i + time_steps][2])
    return np.array(Xs), [np.array(ys_mov_x), np.array(ys_mov_y), np.array(ys_click)]

X, y = load_csv_data()
X_array = np.array(X)
y_array = np.array(y)
time_steps = 20
X_seq, y_seqs = create_sequences(X_array, y_array, time_steps)

inputs = layers.Input(shape=(time_steps, 5)) # Adicionamos a dimensão temporal

# Camada LSTM - Ela processa a sequência e mantém o "estado" interno
x = layers.LSTM(64, return_sequences=False)(inputs) 
x = layers.Dense(64, activation="relu")(x)
# Cabeças de saída (iguais à versão anterior)
mov_x = layers.Dense(1, name="mov_x")(x)
mov_y = layers.Dense(1, name="mov_y")(x)
click = layers.Dense(1, activation="sigmoid", name="click")(x)

model = models.Model(inputs, [mov_x, mov_y, click])

model.compile(
    optimizer="adam",
    loss={
        "mov_x": "mse",
        "mov_y": "mse",
        "click": "binary_crossentropy"
    },
    loss_weights={
        "mov_x": 1.0,
        "mov_y": 1.0,
        "click": 3.0
    }
)

model.summary()

class DebugCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        print(
            f"[EPOCH {epoch+1}] "
            f"mov_x={logs['mov_x_loss']:.4f} | "
            f"mov_y={logs['mov_y_loss']:.4f} | "
            f"click={logs['click_loss']:.4f} | "
            f"total={logs['loss']:.4f}"
        )

model.fit(
    X_seq,
    {
        "mov_x": y_seqs[0],
        "mov_y": y_seqs[1],
        "click": y_seqs[2]
    },
    epochs=15,
    batch_size=64,
    shuffle=True,
    callbacks=[DebugCallback()]
)

model.save("models/mouse_rnn.keras")
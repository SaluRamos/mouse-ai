import tensorflow as tf
from tensorflow.keras import layers, models
import glob
import csv
import numpy as np

def load_csv_data(path="data/data-*.csv"):
    X = []
    y = []
    files = glob.glob(path)
    for file in files:
        print(f"[INFO] lendo: {file}")
        with open(file, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                X.append([
                    float(row["offset_x"]),
                    float(row["offset_y"]),
                    float(row["is_inside_btn"])
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
    print(f"[INFO] amostras carregadas: {X.shape[0]}")
    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")
    return X, y












# Input: offset_x, offset_y, is_inside_btn
# Output: mov_x, mov_y, click
inputs = layers.Input(shape=(3,))

#base comum
x = layers.Dense(128, activation="relu")(inputs)

#cabeça de movimento
x_mov = layers.Dense(64, activation="relu")(x)
mov_x = layers.Dense(1, name="mov_x")(x_mov)
mov_y = layers.Dense(1, name="mov_y")(x_mov)
#cabeça de clique
x_click = layers.Dense(32, activation="relu")(x)
click = layers.Dense(1, activation="sigmoid", name="click")(x_click)

model = models.Model(inputs, [mov_x, mov_y, click])










model.compile(
    optimizer="adam",
    loss={
        "mov_x": "mse",
        "mov_y": "mse",
        "click": "binary_crossentropy"
    },
    loss_weights={
        "mov_x": 2.0,
        "mov_y": 2.0,
        "click": 10.0
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

X, y = load_csv_data()

model.fit(
    X,
    {
        "mov_x": y[:,0],
        "mov_y": y[:,1],
        "click": y[:,2]
    },
    epochs=30,
    batch_size=64,
    shuffle=True,
    callbacks=[DebugCallback()]
)

model.save("models/mouse_mlp.keras")
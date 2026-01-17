#modules
from train_utils import load_csv_data, DebugCallback
#libs
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import LeakyReLU
import numpy as np
#native libs
import time

def create_sequences(X, y, time_steps:int):
    Xs, ys_mov_x, ys_mov_y, ys_click = [], [], [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        # O alvo é sempre o resultado do ÚLTIMO frame da sequência
        ys_mov_x.append(y[i + time_steps][0])
        ys_mov_y.append(y[i + time_steps][1])
        ys_click.append(y[i + time_steps][2])
    return np.array(Xs), [np.array(ys_mov_x), np.array(ys_mov_y), np.array(ys_click)]

X, y = load_csv_data(True, True)
X_array = np.array(X)
y_array = np.array(y)
time_steps = 10
X_seq, y_seqs = create_sequences(X_array, y_array, time_steps)

inputs = layers.Input(shape=(time_steps, 5))

x = layers.LSTM(128, return_sequences=False)(inputs)
x = layers.Dense(64, activation="relu")(x)
mov_x = layers.Dense(1, name="mov_x")(x)
mov_y = layers.Dense(1, name="mov_y")(x)

model = models.Model(inputs, [mov_x, mov_y])

model.compile(
    optimizer="adam",
    loss={
        "mov_x": "huber",
        "mov_y": "huber",
    },
    loss_weights={
        "mov_x": 3.0,
        "mov_y": 3.0,
    }
)

early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

model.summary()

#continuar treinamento do modelo anterior
# model = load_model("models/mouse_rnn_300e.keras")
batch_size = 2**13
print(f"batch_size={batch_size}")
time.sleep(5)

model.fit(
    X_seq,
    {
        "mov_x": y_seqs[0],
        "mov_y": y_seqs[1]
    },
    epochs=170,
    batch_size=batch_size,
    shuffle=False,
    callbacks=[DebugCallback(), early_stop]
)

model.save("models/mouse_rnn.keras")
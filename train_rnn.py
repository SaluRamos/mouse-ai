from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from train_utils import load_csv_data, DebugCallback
from enums import ModelType

def create_sequences(X, y, time_steps:int):
    Xs, ys_mov_x, ys_mov_y, ys_click = [], [], [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        # O alvo é sempre o resultado do ÚLTIMO frame da sequência
        ys_mov_x.append(y[i + time_steps][0])
        ys_mov_y.append(y[i + time_steps][1])
        ys_click.append(y[i + time_steps][2])
    return np.array(Xs), [np.array(ys_mov_x), np.array(ys_mov_y), np.array(ys_click)]

X, y = load_csv_data(ModelType.RNN)
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
        "click": 5.0
    }
)

early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

model.summary()

model.fit(
    X_seq,
    {
        "mov_x": y_seqs[0],
        "mov_y": y_seqs[1],
        "click": y_seqs[2]
    },
    epochs=120,
    batch_size=64,
    shuffle=True,
    callbacks=[DebugCallback(), early_stop]
)

model.save("models/mouse_rnn.keras")
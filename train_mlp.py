#modules
from train_utils import load_csv_data, DebugCallback
#libs
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
#native libs
import time

inputs = layers.Input(shape=(5,))
#base comum
x = layers.Dense(64, activation="relu")(inputs)
#cabeça de movimento
x_mov = layers.Dense(64, activation="selu")(x)
mov_x = layers.Dense(1, name="mov_x")(x_mov)
mov_y = layers.Dense(1, name="mov_y")(x_mov)
x_click = layers.Dense(1, activation="relu")(x)
model = models.Model(inputs, [mov_x, mov_y])

model.compile(
    optimizer="adam",
    loss={
        "mov_x": "mse",
        "mov_y": "mse",
    },
    loss_weights={
        "mov_x": 3.0,
        "mov_y": 3.0,
    }
)

early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

model.summary()

X, y = load_csv_data(True, True)

#continuar treinamento do modelo anterior
# model = load_model("models/mouse_rnn_300e.keras")
batch_size = 16
print(f"batch_size={batch_size}")
time.sleep(5)

model.fit(
    X,
    {
        "mov_x": y[:,0],
        "mov_y": y[:,1]
    },
    epochs=50, #10 is low, 20 is ok
    batch_size=batch_size,
    shuffle=True,
    callbacks=[DebugCallback(), early_stop]
)

model.save("models/mouse_mlp.keras")
#modules
from train_utils import load_csv_data, DebugCallback, create_sequences
#libs
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

#offset_x, offset_w, btn_w, btn_h
inputs = layers.Input(shape=(4,))
#base comum
x = layers.Dense(64, activation="relu")(inputs)
x_mov = layers.Dense(64, activation="selu")(x)
#cabeça de movimento
mov_x = layers.Dense(1, name="mov_x")(x_mov)
mov_y = layers.Dense(1, name="mov_y")(x_mov)
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
# model = load_model("models/mouse_mlp.keras")

model.fit(
    X,
    {
        "mov_x": y[:,0],
        "mov_y": y[:,1]
    },
    epochs=50, #10 is low, 20 is ok
    batch_size=16,
    shuffle=True,
    callbacks=[DebugCallback(), early_stop]
)

model.save("models/mouse_mlp.keras")
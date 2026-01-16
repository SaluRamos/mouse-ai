from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from train_utils import load_csv_data, DebugCallback
from enums import ModelType

inputs = layers.Input(shape=(5,))
#base comum
x = layers.Dense(64, activation="relu")(inputs)
#cabeça de movimento
x_mov = layers.Dense(64, activation="selu")(x)
mov_x = layers.Dense(1, name="mov_x")(x_mov)
mov_y = layers.Dense(1, name="mov_y")(x_mov)
x_click = layers.Dense(1, activation="relu")(x)
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
        "mov_x": 1.0,
        "mov_y": 1.0,
        "click": 5.0
    }
)

early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

model.summary()

X, y = load_csv_data(ModelType.MLP)

model.fit(
    X,
    {
        "mov_x": y[:,0],
        "mov_y": y[:,1],
        "click": y[:,2]
    },
    epochs=20, #10 is low, 20 is ok
    batch_size=64,
    shuffle=True,
    callbacks=[DebugCallback(), early_stop]
)

model.save("models/mouse_mlp.keras")
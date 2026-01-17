#modules
from utils import calc_degree_btw_vecs, calc_vec_magnitude
#libs
import tensorflow as tf
import numpy as np
#native libs
import csv
import glob

def load_csv_data(filter_mov0:bool, filter_big_degree:bool):
    X = []
    y = []
    path = "data/data-*.csv"
    files = glob.glob(path)
    removed_lines_with_low_mov = 0
    removed_lines_with_big_degree = 0
    rows_with_click = 0
    for file in files:
        print(f"LENDO '{file}'")
        with open(file, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                #penalizar cursor parado
                if filter_mov0 and float(row["mov_x"]) == 0 and float(row["mov_y"]) == 0:
                    removed_lines_with_low_mov += 1
                    continue
                #penalizar direção oposta ao alvo
                if filter_big_degree and calc_degree_btw_vecs([float(row["mov_x"]), float(row["mov_y"])], [float(row["offset_x"]), float(row["offset_y"])]) > 120.0:
                    removed_lines_with_big_degree += 1
                    continue
                if int(row["click"]) == 1:
                    rows_with_click += 1
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
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    if filter_mov0:
        print(f"REMOVED LINES WITH LOW MOV = {removed_lines_with_low_mov}")
    if filter_big_degree:
        print(f"REMOVED LINES WITH BIG DEGREE = {removed_lines_with_big_degree}")
    print(f"ROWS WITH CLICK = {rows_with_click}")
    print(f"TOTAL ROWS: {X.shape[0]}")
    return X, y

class DebugCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        print(
            f"[EPOCH {epoch+1}] "
            f"total={logs['loss']:.4f}"
            f"mov_x={logs['mov_x_loss']:.4f} | "
            f"mov_y={logs['mov_y_loss']:.4f} | "
        )

print(tf.config.list_physical_devices('GPU'))
print("Cuda Disponível:", tf.test.is_built_with_cuda())
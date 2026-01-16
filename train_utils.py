#libs
import tensorflow as tf
import numpy as np
#native libs
import math
import csv
import glob
from enums import ModelType

def calc_vec_magnitude(v:list) -> float:
    return math.sqrt(v[0]**2 + v[1]**2)

def calc_degree_btw_vecs(v1:list, v2:list) -> float:
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    if mag1 == 0 or mag2 == 0: return 0
    cos_theta = max(-1, min(1, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_theta))

def load_csv_data(model_type:ModelType):
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
                if model_type == ModelType.MLP:
                    #não remover clicks
                    if not int(row["click"]) == 1:
                        #penalizar cursor parado
                        #escala: 1 = 800 px, 0.1 = 80px, 0.01 = 8px, 0.005 = 4px
                        if calc_vec_magnitude([float(row["mov_x"]), float(row["mov_y"])]) < 0.0025:
                            removed_lines_with_low_mov += 1
                            continue
                        #penalizar direção oposta ao alvo
                        if calc_degree_btw_vecs([float(row["mov_x"]), float(row["mov_y"])], [float(row["offset_x"]), float(row["offset_y"])]) > 120.0:
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
    if model_type == ModelType.MLP:
        print(f"REMOVED LINES WITH LOW MOV = {removed_lines_with_low_mov}")
        print(f"REMOVED LINES WITH BIG DEGREE = {removed_lines_with_big_degree}")
    print(f"ROWS WITH CLICK = {rows_with_click}")
    print(f"TOTAL ROWS: {X.shape[0]}")
    return X, y

class DebugCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        print(
            f"[EPOCH {epoch+1}] "
            f"mov_x={logs['mov_x_loss']:.4f} | "
            f"mov_y={logs['mov_y_loss']:.4f} | "
            f"click={logs['click_loss']:.4f} | "
            f"total={logs['loss']:.4f}"
        )
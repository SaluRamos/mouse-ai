#modules
from collect_paths import plot_path
#libs
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
#native libs
import csv
import glob
import os
import math

MAX_SEQ_LEN = 100 # Baseado em analises
FEATURE_DIM = 2 # mov_x, mov_y
COND_DIM = 4 # offset_x, offset_y, btn_w, btn_h (condições iniciais)

def load_padded_sequences():
    """
    Carrega os dados agrupados por 'path' (caminho até o clique).
    Retorna:
      - sequences: (N, MAX_SEQ_LEN, 2) -> Sequências de movimentos (mov_x, mov_y)
      - conditions: (N, 4) -> Estado inicial do caminho (distância alvo, tamanho botão)
    """
    path = "data/data-*.csv"
    if not os.path.exists("data/"):
        path = "../" + path
    files = glob.glob(path)
    all_sequences = []
    all_conditions = []
    for file in files:
        with open(file, newline="") as f:
            reader = csv.DictReader(f)
            current_path = []
            initial_condition = None 
            for row in reader:
                # Dados do tick atual
                mov = [float(row["mov_x"]), float(row["mov_y"])]
                if initial_condition is None:
                    initial_condition = [
                        float(row["offset_x"]),
                        float(row["offset_y"]),
                        float(row["btn_w"]),
                        float(row["btn_h"])
                    ]
                current_path.append(mov)
                if int(row["click"]) == 1:
                    # Filtra paths muito curtos (ruído) ou muito longos
                    if len(current_path) > 5 and len(current_path) <= MAX_SEQ_LEN:
                        # Padding: Preenche o restante com (0,0) até chegar em MAX_SEQ_LEN
                        # Isso ensina a rede que o movimento deve "parar"
                        pad_len = max(MAX_SEQ_LEN - len(current_path), 0)
                        padded_path = current_path + [[0.0, 0.0]] * pad_len
                        all_sequences.append(padded_path)
                        all_conditions.append(initial_condition)
                    current_path = []
                    initial_condition = None
    X_seq = np.array(all_sequences, dtype=np.float32)
    X_cond = np.array(all_conditions, dtype=np.float32)
    print(f"Sequências carregadas: {X_seq.shape}")
    print(f"Condições carregadas: {X_cond.shape}")
    return X_seq, X_cond

class GANMonitor(tf.keras.callbacks.Callback):

    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        # Criamos uma condição fixa para ver como o mesmo movimento evolui
        # Ex: Alvo em 0.5, 0.5 com botão de tamanho padrão
        self.test_cond = np.array([[0.5, 0.5, 0.05, 0.03]], dtype=np.float32)
        self.plots_path = "plots"
        if not os.path.exists("models"):
            self.plots_path = "../" + self.plots_path
        if not os.path.exists(self.plots_path):
            os.makedirs(self.plots_path)

    def on_epoch_end(self, epoch, logs=None):
        generator_new_epoch_value = self.model.generator.epoch_tracker.read_value() + epoch
        discriminator_new_epoch_value = self.model.discriminator.epoch_tracker.read_value() + epoch
        if (epoch + 1) % 1 == 0:
            off_x, off_y, bw, bh = self.test_cond[0]
            noise = tf.random.normal(shape=(1, self.latent_dim))
            generated = self.model.generator([noise, self.test_cond]).numpy()[0]
            path_x = np.cumsum(generated[:, 0])
            path_y = np.cumsum(generated[:, 1])
            btn_x = off_x - bw/2
            btn_y = off_y - bh/2
            rect = Rectangle((btn_x, btn_y), bw, bh, fill=False, color='red', linewidth=2)
            plt.gca().add_patch(rect)
            plt.plot(path_x, path_y, '-o', markersize=2, alpha=0.6)
            plt.scatter(path_x[-1], path_y[-1], color="#ce690c", s=30, edgecolors='white', zorder=5)
            all_x = np.concatenate([path_x, [0, btn_x, btn_x + bw]])
            all_y = np.concatenate([path_y, [0, btn_y, btn_y + bh]])
            margin = 0.05
            plt.xlim(min(all_x) - margin, max(all_x) + margin)
            plt.ylim(min(all_y) - margin, max(all_y) + margin)
            plt.gca().set_aspect("equal")
            plt.title(f"Epoch {epoch+1} | acc real: {logs['acc_real']:.4f} | g hit rate: {logs['g_hit']:.4f}")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.savefig(f"{self.plots_path}/epoch_g_{generator_new_epoch_value}_d_{discriminator_new_epoch_value}.png")
            plt.close()
        print(f"\nGENERATOR EPOCH = {generator_new_epoch_value}, DISCRIMINATOR EPOCH = {discriminator_new_epoch_value}")

print(tf.config.list_physical_devices('GPU'))
print("Cuda Disponível:", tf.test.is_built_with_cuda())
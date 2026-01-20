#modules
from utils import calc_degree_btw_vecs, calc_vec_magnitude, print_sorted_dict, get_base_path
#libs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats
#native libs
import glob
import csv
import os

def collect_paths(plot:bool=False) -> list:
    path = f"{get_base_path()}data/good/data-*.csv"
    files = glob.glob(path)
    paths = []
    actual_path = []
    for file in files:
         with open(file, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if int(row["click"]) == 1 and len(actual_path) > 0:
                    paths.append(actual_path.copy())
                    if plot:
                        plot_path(actual_path, True)
                    actual_path.clear()
                global_mouse_pos = [float(row["global_pos_x"]), float(row["global_pos_y"])]
                window_size = [int(row["window_width"]), int(row["window_height"])]
                root = [int(row["root_x"]), int(row["root_y"])]
                local_mouse_pos = [(global_mouse_pos[0] - root[0])/window_size[0], 
                                   (global_mouse_pos[1] - root[1])/window_size[1]]
                mov_vec = [float(row["mov_x"]), float(row["mov_y"])]
                offset = [float(row["offset_x"]), float(row["offset_y"])]
                btn_size = [float(row["btn_w"]), float(row["btn_h"])]
                btn_pos = [local_mouse_pos[0] + offset[0] - btn_size[0]/2,
                           local_mouse_pos[1] + offset[1] - btn_size[1]/2]
                degree_diff = calc_degree_btw_vecs(mov_vec, offset)
                tick_info = {
                    "pos":local_mouse_pos,
                    "offset":offset,
                    "btn_size":btn_size,
                    "btn_pos":btn_pos,
                    "mov":mov_vec,
                    "is_inside_btn":row["is_inside_btn"],
                    "degree_diff":degree_diff
                }
                actual_path.append(tick_info)
    
    print(f"TOTAL PATHS = {len(paths)}")
    return paths

def plot_path(path: list[dict], show:bool, save_path:str="") -> None:
    xs = [tick["pos"][0] for tick in path]
    ys = [tick["pos"][1] for tick in path]

    plt.figure()
    plt.plot(xs, ys)

    plt.scatter(xs[0], ys[0], label="start")
    plt.scatter(xs[-1], ys[-1], label="end")

    btn_x = path[-1]["btn_pos"][0]
    btn_y = path[-1]["btn_pos"][1]
    bw = path[-1]["btn_size"][0]
    bh = path[-1]["btn_size"][1]

    rect = Rectangle(
        (btn_x, btn_y),
        bw, bh,
        fill=False
    )
    plt.gca().add_patch(rect)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal", adjustable="box")

    if show:
        plt.show()
    if not save_path == "":
        plt.savefig(save_path)
        plt.close()

def plot_ticks_per_path(paths:list) -> None:
    data = {}
    for path in paths:
        ticks = len(path)
        if ticks not in data.keys():
            data[ticks] = 0
        data[ticks] += 1
    print_sorted_dict(data)
    values = []
    for ticks, incidents in data.items():
        values.extend([ticks] * incidents)
    params = stats.gamma.fit(values)
    #plotar colunas
    xs = np.array(sorted(data.keys()))
    ys = np.array([data[k] for k in xs])
    plt.bar(xs, ys)
    #plotar curva
    xx = np.linspace(xs.min(), xs.max(), 500)
    pdf = stats.gamma.pdf(xx, *params)
    pdf_scaled = pdf*len(values)
    plt.plot(xx, pdf_scaled, linewidth=2, color="#cf8e00")
    plt.show()

if __name__ == "__main__":
    paths = collect_paths(False)
    plot_ticks_per_path(paths)

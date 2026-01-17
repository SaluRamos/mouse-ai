#modules
from utils import calc_degree_btw_vecs, calc_vec_magnitude
#libs
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
#native libs
import glob
import csv

def collect_paths() -> None:
    path = "data/data-*.csv"
    data = {}
    files = glob.glob(path)

    paths = []
    actual_path = []

    for file in files:
         with open(file, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if int(row["click"]) == 1 and len(actual_path) > 0:
                    paths.append(actual_path.copy())
                    plot_path(actual_path)
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
                    "is_inside_btn":row["is_inside_btn"]
                }
                actual_path.append(tick_info)
    
    print(f"TOTAL PATHS = {len(paths)}")

def plot_path(path: list[dict]) -> None:
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

    plt.show()

if __name__ == "__main__":
    collect_paths()
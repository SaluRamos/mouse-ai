import math
import ctypes
import os
from pathlib import Path
import json

def get_best_model() -> str:
    base_path = Path(f"{get_base_path()}models/")
    folders = {}
    best_folder = ""
    best_score = -float('inf')
    for folder in base_path.iterdir():
        if not folder.is_dir():
            continue
        info_path = folder / "info.json"
        if not info_path.exists():
            continue
        try:
            with open(info_path, "r") as f:
                stats = json.load(f)
            g_hit = stats.get("g_hit", 0.0)
            acc_real = stats.get("acc_real", 0.0)
            acc_fake = stats.get("acc_fake", 0.0)
            score = (0.5 - acc_real) + (0.5 - acc_fake) + g_hit
            folders[folder.absolute()] = [score, g_hit, acc_real, acc_fake]
            if score > best_score:
                best_score = score
                best_folder = str(folder.absolute())
        except Exception as e:
            print(f"Erro ao processar pasta {folder}: {e}")

    print("\n--- Ranking de Modelos (Score) ---")
    sorted_folders = sorted(folders.items(), key=lambda item: item[1], reverse=False)
    for name, s in sorted_folders: # Mostra o Top 10
        print(f"Epoch: {name} | score={s[0]:.4f}, g_hit={s[1]:.4f}, acc_real={s[2]:.4f}, acc_fake={s[3]:.4f}")
    print("----------------------------------\n")
    return best_folder
    

def get_base_path() -> str:
    if not os.path.exists("models/"):
        return "../"
    return ""

def calc_vec_magnitude(v:list) -> float:
    return math.sqrt(v[0]**2 + v[1]**2)

def calc_degree_btw_vecs(v1:list, v2:list) -> float:
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    if mag1 == 0 or mag2 == 0: return 0
    cos_theta = max(-1, min(1, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_theta))

def print_sorted_dict(data) -> None:
    for key in sorted(data.keys()):
        print(f"key = {key}, value = {data[key]}")

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

def get_mouse_pos() -> tuple[int, int]:
    pt = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

SendInput = ctypes.windll.user32.mouse_event
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

def mov_mouse(mov_x:int, mov_y:int) -> None:
    SendInput(MOUSEEVENTF_MOVE, mov_x, mov_y, 0, 0)

def click_mouse() -> None:
    SendInput(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    SendInput(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
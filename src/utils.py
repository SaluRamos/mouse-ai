import math
import ctypes

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
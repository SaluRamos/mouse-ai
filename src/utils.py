import math

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
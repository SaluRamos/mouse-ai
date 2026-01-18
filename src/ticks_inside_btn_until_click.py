#modules
from utils import print_sorted_dict
#libs
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
#native libs
import glob
import csv
import random

def get_ticks_inside_btn_until_click() -> None:
    path = "data/data-*.csv"
    data = {}
    files = glob.glob(path)
    for file in files:
        with open(file, newline="") as f:
            reader = csv.DictReader(f)
            counting = False
            start_i = 0
            for i, row in enumerate(reader):
                #entrou no botão
                if int(row["is_inside_btn"]) == 1 and not counting:
                    start_i = i
                    counting = True
                #saiu do botão (cancela contagem)
                if int(row["is_inside_btn"]) == 0 and counting:
                    counting = False
                #click (finaliza contagem)
                if int(row["click"]) == 1:
                    counting = False
                    ticks = i - start_i
                    if ticks not in data:
                        data[ticks] = 0
                    data[ticks] += 1
    #CLARAMENTE UMA DISTRIBUIÇÃO!!!
    print_sorted_dict(data)
    #remover outliers
    MAX_TICK = 24 # visualizando os dados considero que valores acima disso são outliers
    for key in list(data.keys()):
        if key > MAX_TICK:
            del data[key]
    #valores expandidos
    values = []
    for ticks, incidents in data.items():
        values.extend([ticks] * incidents)
    #encontra parametros da curva que parece com os dados
    params = stats.gamma.fit(values)
    print(f"{params}")
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

def random_ticks_inside_btn(shape:float=10.53369, loc:float=-0.93502, scale:float=0.80039) -> int:
    r = random.gammavariate(shape, scale)
    r += loc
    return max(0, round(r))

if __name__ == "__main__":
    get_ticks_inside_btn_until_click()
    print("----------------")
    print("Gerando numeros aleatorios com base na distribuição")
    data = {}
    for i in range(100000):
        value = random_ticks_inside_btn()
        if value not in data:
            data[value] = 0
        data[value] += 1
    print_sorted_dict(data)

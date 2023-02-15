import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

import numpy as np
import alphashape
from descartes import PolygonPatch

import concurrent.futures
import time

colores = ['blue', 'red', 'green']
archivo = "data/animacion.txt"

N = 10000
L = 475

np_steps = 1000

fig, ax = plt.subplots()

def generate_data(start_frame, num_frames):
    for i in range(start_frame, num_frames, 50):
        num = start_frame + i
        x = np.loadtxt(archivo, usecols=0, skiprows=num * N, max_rows=N)
        y = np.loadtxt(archivo, usecols=1, skiprows=num * N, max_rows=N)
        estado = np.loadtxt(archivo, usecols=3, skiprows=num * N, max_rows=N, dtype=int)

        yield num, x, y, estado

def update(num, x, y, estado):
    points = np.column_stack((x, y))
    mask = np.where(estado > 0)

    plt.cla()

    concave_flag = False
    try:
        alpha = alphashape.optimizealpha(points[mask])

        hull = alphashape.alphashape(points[mask], alpha)
        hull_exterior_points = np.array(hull.exterior.coords)
        concave_flag = True

        with open('area.txt', 'a') as file:
            file.write(f"{round(hull.area, 1)} {num} {alpha} \n")

        print(f"(Area, alpha, time) = ({hull.area}, {alpha}, {num})")

    except:
        print(f"Not enough points {num}")

    plt.xlabel("x coordinate")
    plt.ylabel("y coordinate")

    plt.axis('square')
    plt.grid()
    plt.xlim(-1, L+1)
    plt.ylim(-1, L+1)

    if concave_flag:
        ax.add_patch(PolygonPatch(hull, fill=False, color='black', linewidth=0.8))

    for j in range(N):
        circ = patches.Circle((x[j], y[j]), 1, alpha=0.7, fc=colores[estado[j]])
        ax.add_patch(circ)

    plt.savefig(f"video/pic{num}.png", dpi=100)


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = []

        for num, x, y, estado in generate_data(0, np_steps):
            results.append(executor.submit(update, num, x, y, estado))

        for result in concurrent.futures.as_completed(results):
            try:
                result.result()
            except Exception as e:
                print(f"Exception occurred: {e}")
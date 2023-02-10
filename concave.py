import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

from joblib import Parallel, delayed
import time

import numpy as np
import alphashape
from descartes import PolygonPatch

colores = ['blue', 'red', 'green']
archivo = "data/animacion.txt"

N = 1000
L = 150
np_steps = 500

fig, ax = plt.subplots()

def update(num):
    x = np.loadtxt(archivo, usecols=0, skiprows=num * N, max_rows=N)
    y = np.loadtxt(archivo, usecols=1, skiprows=num * N, max_rows=N)
    estado = np.loadtxt(archivo, usecols=3, skiprows=num * N, max_rows=N, dtype=int)
    
    points = np.column_stack((x, y))
    mask = np.where(estado>0)
    
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
    plt.xlim(-1,L+1)
    plt.ylim(-1,L+1)

    if concave_flag:
        #ax.scatter(hull_pts[:,0], hull_pts[:,1], color='red')
        ax.add_patch(PolygonPatch(hull, fill=False, color='black', linewidth=0.8))

    for j in range(N):
        circ = patches.Circle((x[j],y[j]), 1, alpha=0.7, fc= colores[estado[j]])
        ax.add_patch(circ)
    
    plt.savefig(f"video/pic{num}.png", dpi=100)


# Parallel loop for calculations, uso ALL CPU. n_jobs control #threads:
Parallel(n_jobs=-1)(delayed(update)(num) for num in range(0, np_steps, 5))

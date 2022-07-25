import os
import numpy as np
 
alphas = np.linspace(0.1,1.0,5)
l1_ratiios = np.linspace(0.1,1.0,5)

for p1 in alphas:
    for p2 in l1_ratiios:
        print(f"loggign experiment for alpha={p1} and l1_ratio={p2}")
        os.system(f"python demo_ml.py -a={p1} -l1={p2}")

import os
from re import L
 
p1s = range(0,3)
p2s = range(0,3)

for p1 in p1s:
    for p2 in p2s:
        print(f"loggign experiment for p1={p1} and p2={p2}")
        os.system(f"python demo.py -p1={p1} -p2={p2}")

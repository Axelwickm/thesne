import abc
import numpy as np
from importlib import util
import random
import bisect

import os

from thesne.model.dynamic_tsne import dynamic_tsne
from thesne.examples import plot

def generate_data(n, dt):
    """ Generate n entries over dt days """
    data = []

    for i in range(n):
        p = {}
        t = dt*random.random()
        p["id"] = i
        p["time"] = t
        p["analyzed_data"] = [random.random() * 0.5, 0.8, random.random() * 0.2, t / dt, (3 * t / dt * t / dt * t / dt - 0.5 * t / dt * t / dt - t / dt + 2)/5]
        data.append(p)

    return data


def convert_to_Xs(data, time_window, steps, max_time=None):
    if max_time is None:
        max_time = max(data, key = lambda x: x["time"])["time"]
    step_size = max_time / steps
    data = sorted(data, key = lambda x: x["time"])
    times = [x["time"] for x in data]
    
    Xs = []
    IDs = []
    for tmin in np.arange(0.0, max_time, step_size):
        tmax = tmin + time_window
        left = bisect.bisect_right(times, tmin)
        right = bisect.bisect_left(times, tmax)
        
        x = []
        ids = []
        for entry in data[left:right]:
            x.append(entry["analyzed_data"])
            ids.append(entry["id"])
            
        #print(str(len(x))+"   "+str(tmin)+"  ->  "+str(tmax))
        Xs.append(np.array(x))
        IDs.append(ids)
    return Xs, IDs
   
def main():
    seed = 0
    
    data = generate_data(200, 100)
    
    # FIXME: This can create empty Xs, which crashes the program
    Xs, IDs = convert_to_Xs(data, 30, 5, max_time=100)  
    
    Ys = dynamic_tsne(Xs, IDs, perplexity=30, n_epochs=2000, initial_lr=150, final_lr=50, lmbda=0.1, verbose=1, sigma_iters=50,
                      initial_momentum = 0.3, random_state=seed)
        
    for Y in Ys:
        plot.plot(Y)

if __name__ == "__main__":
    main()
    

  
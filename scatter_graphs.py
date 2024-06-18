import torch
from scatter import Scatter
from tqdm import tqdm
import pickle as pk
from multiprocessing import Pool, cpu_count, Manager
from functools import partial

def scatter_graph(graph):
    in_channels = graph.x.size(0)
    max_graph_size = graph.x.size(0)
    scattering = Scatter(in_channels, max_graph_size)
    scatter_coeffs = scattering(graph)
    return scatter_coeffs

def write_scatter_coeffs(scatter_coeffs, lock, fname):
    lock.acquire()
    with open(fname, 'ab') as f:
        pk.dump(scatter_coeffs, f)
    lock.release()

def process_scatter(data, out_fname, lock):
    scatter_coeffs = scatter_graph(data)
    write_scatter_coeffs(scatter_coeffs, lock, out_fname)

def run_scattering_pkl():
    manager = Manager()
    write_lock = manager.Lock()
    out_fname = 'scatter_coeffs_new.pkl'

    with Pool(processes=cpu_count()) as pool:
        with open('graph_data.pkl', 'rb') as f:
            for _ in tqdm(range(2465)):
                graph = pk.load(f)
                partial_process_scatter = partial(process_scatter, 
                                                  data=graph, 
                                                  out_fname=out_fname, 
                                                  lock=write_lock)
                pool.apply_async(partial_process_scatter)
            pool.close()
            pool.join()

if __name__ == '__main__':
    run_scattering_pkl()

import numpy as np
import dask.array as da

def fft(x, memsize=128*1024**2):
    x = da.from_array(x, chunks=memsize)
    N = x.size
    R = memsize
    C = N//R
    x = x.reshape((R, N//R)).T
    

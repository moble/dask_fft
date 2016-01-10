def DAFT(x, axis=-1, chunksize=2**26):
    """Disk-Array Fourier Transform
    
    This function enables Fourier transforms of a very large series, where the entire series will not fit in memory.
    The standard radix-2 Cooley–Tukey algorithm is used to split the series up into smaller pieces until a given
    piece can be done entirely in memory.  This smaller result is then stored as a `dask.array`, and combined with
    other similar results out of memory, using dask.
    
    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is used.
    chunksize : int, optional
        Chunksize to use when splitting up the input array.  Default is 2**24,
        which is about 64MB -- a reasonable target that reduces memory usage.
    
    """
    import numpy as np
    import dask.array as da

    N = x.shape[axis]
    if isinstance(x, da.Array):
        x_da = x.rechunk(chunks={axis: chunksize})
    else:
        x_da = da.from_array(x, chunks={axis: chunksize})

    W = np.exp(-2j * np.pi * np.arange(N) / N)

    if len(x_da.chunks[axis]) != 1:
        # TODO: Fix the following lines to be correct when x is multi-dimensional
        FFT_even = DAFT(x_da[::2], axis, chunksize=chunksize)
        FFT_odd = DAFT(x_da[1::2], axis, chunksize=chunksize)
    else:
        # TODO: Fix the following lines to be correct when x is multi-dimensional
        FFT_even = da.fft.fft(x_da[::2], n=None, axis=axis)
        FFT_odd = da.fft.fft(x_da[1::2], n=None, axis=axis)

    # TODO: Fix the following line to broadcast W correctly when x is multi-dimensional
    return da.concatenate([FFT_even + W[:N//2] * FFT_odd, FFT_even + W[N//2:] * FFT_odd], axis=axis)

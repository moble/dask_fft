import sys

def fft(x, axis=-1, chunksize=2**26, available_memory=(4 * 1024**3), cache=None):
    """Simple wrapper for DAFT FFT function

    This function calls the DAFT function, but also performs the computation of
    the FFT, and returns the result as a numerical array.

    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is used.
    chunksize : int, optional
        Chunksize to use when splitting up the input array.  Default is 2**24,
        which is about 64MB -- a reasonable target that reduces memory usage.
    available_memory : int, optional
        Maximum amount of RAM to use for caching during computation.  Defaults
        to 4*1024**3, which is 4GB.

    """
    if cache is None:
        from chest import Chest  # For more flexible caching
        cache = Chest(available_memory=available_memory)
    X_dask = DAFT(x, axis=axis, chunksize=chunksize)
    return X_dask.compute(cache=cache)

def fft_to_hdf5(x, filename, axis=-1, chunksize=2**26, available_memory=(4 * 1024**3), cache=None):
    """Simple wrapper for DAFT FFT function that writes to HDF5

    This function calls the DAFT function, but also performs the computation of
    the FFT, and outputs the result into the requested HDF5 file

    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    filename : string
        Relative or absolute path to HDF5 file.  If this string contains a
        colon, the preceding part is taken as the filename, while the following
        part is taken as the dataset group name.  The default group name is 'X'.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is used.
    chunksize : int, optional
        Chunksize to use when splitting up the input array.  Default is 2**24,
        which is about 64MB -- a reasonable target that reduces memory usage.
    available_memory : int, optional
        Maximum amount of RAM to use for caching during computation.  Defaults
        to 4*1024**3, which is 4GB.

    """
    from h5py import File
    from dask import set_options
    from dask.array import store
    if cache is None:
        from chest import Chest  # For more flexible caching
        cache = Chest(available_memory=available_memory)
    if ':' in filename:
        filename, groupname = filename.split(':')
    else:
        groupname = 'X'
    X_dask = DAFT(x, axis=axis, chunksize=chunksize)
    with set_options(cache=cache):
        with File(filename, 'w') as f:
            output = f.create_dataset(groupname, shape=X_dask.shape, dtype=X_dask.dtype)
            store(X_dask, output)
    return

def DAFT(x, axis=-1, chunksize=2**26):
    """Disk-Array Fourier Transform
    
    This function enables Fourier transforms of a very large series, where the
    entire series will not fit in memory.  The standard radix-2 Cooley–Tukey
    algorithm is used to split the series up into smaller pieces until a given
    piece can be done entirely in memory.  This smaller result is then stored
    as a `dask.array`, and combined with other similar results out of memory,
    using dask.
    
    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is used.
    chunksize : int, optional
        Chunksize to use when splitting up the input array.  Default is 2**24,
        which is about 64MB -- a reasonable target that reduces memory usage.

    Returns
    -------
    X_da : dask Array object
        The Fourier transform is not yet computed; you must call
        `X_da.compute()` on the result to perform the computation.

    Example
    -------
    >>> import numpy as np
    >>> from chest import Chest  # For more flexible caching
    >>> cache = Chest(available_memory=(4 * 1024**3))  # Use 4GB at most
    >>> N = 2**26
    >>> chunksize = N//(2**2)
    >>> np.random.seed(1234)
    >>> x = np.random.random(N) + 1j*np.random.random(N)
    >>> X_dask = DAFT(x, chunksize=chunksize)
    >>> %tic
    >>> X_DAFT = X_dask.compute(cache=cache)
    >>> %toc
    >>> %tic
    >>> X_np = np.fft.fft(x)
    >>> %toc
    >>> np.allclose(X_DAFT, X_np)

    """
    import numpy as np
    import dask.array as da
    
    if axis<0:
        axis = x.ndim + axis
    N = x.shape[axis]
    chunks = tuple(1 if ax!=axis else chunksize for ax,dim in enumerate(x.shape))
    if isinstance(x, da.Array):
        x_da = x.rechunk(chunks=chunks)
    else:
        x_da = da.from_array(x, chunks=chunks)

    W = np.exp(-2j * np.pi * np.arange(N) / N)

    # print(x.shape, axis, x_da.chunks, x_da.chunks[axis]); sys.stdout.flush()
    slice_even = tuple(slice(None) if ax!=axis else slice(None, None, 2) for ax in range(x_da.ndim))
    slice_odd =  tuple(slice(None) if ax!=axis else slice(1, None, 2)    for ax in range(x_da.ndim))
    if len(x_da.chunks[axis]) != 1:
        # TODO: Fix the following lines to be correct when x is multi-dimensional
        FFT_even = DAFT(x_da[slice_even], axis, chunksize=chunksize)
        FFT_odd = DAFT(x_da[slice_odd], axis, chunksize=chunksize)
    else:
        # TODO: Fix the following lines to be correct when x is multi-dimensional
        FFT_even = da.fft.fft(x_da[slice_even], n=None, axis=axis)
        FFT_odd = da.fft.fft(x_da[slice_odd], n=None, axis=axis)

    # TODO: Fix the following line to broadcast W correctly when x is multi-dimensional
    return da.concatenate([FFT_even + W[:N//2] * FFT_odd, FFT_even + W[N//2:] * FFT_odd], axis=axis)

# dask_fft
Simple code for out-of-core FFTs

Though the FFT algorithm is brilliant---both extremely fast and efficient---it
is still limited by the size of the computer's memory.  But we can extend the
standard Cooley-Tukey algorithm to work on pieces of the full data set, storing
most of it on disk, then stepping through and combining those pieces to get the
final result.

The need for this sort of thing arises, For example, with the Advanced LIGO
gravitational-wave observatory.  This instrument is sensitive to frequencies as
low as 10 Hz, which requires extremely long time-domain waveforms.  Also, the
peak frequencies for low-mass systems are around 4096 Hz, which means that
those long waveforms must be finely sampled.  This can easily result in
waveforms requiring memory in the GB range.  If you need a few of these and
their FFTs, you'll quickly run out of memory.

Fortunately, a module has been built for python which allows relatively
automatic out-of-core computations.  The
[`dask` module](http://dask.pydata.org/en/latest/) does this.

My approach is very simplistic; I simply divide the data in half until my
pieces are small enough to FFT in memory.  I then use `dask` to combine these
pieces as per the standard Cooley-Tukey algorithm.  There are likely much
better ways to do this.  See, for example,
[this paper](http://link.springer.com/chapter/10.1007/978-1-4612-1516-5_14).

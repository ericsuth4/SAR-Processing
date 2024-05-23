import numpy as np
from scipy import signal
from scipy.signal import chirp
import multiprocessing
import time
import os
from functools import partial
from multiprocessing import Pool

import functions as fun



numworkers = os.cpu_count()


#####  Range Compression  ######

def compress_chunk(data_chunk, sinChirp, cosChirp):
    compressed_chunk = np.zeros_like(data_chunk, dtype=np.complex64)
    for i in range(data_chunk.shape[0]):
        sin_corr = signal.correlate(data_chunk[i], 1j*sinChirp, mode='same', method='fft')
        cos_corr = signal.correlate(data_chunk[i], cosChirp, mode='same', method='fft')
        compressed_chunk[i] = cos_corr + sin_corr

    return compressed_chunk

def compress(data, tp, fs, fl, fh, direction, window=True, plot=False, num_processes=numworkers):

    ref_samp = int(tp*fs)  ##  finding number of samples that belong in the reference chirp
    t_ref = np.linspace(0, tp, ref_samp)
    f_ref = np.linspace(-fs/2, fs/2, ref_samp)
    print(f'Chirp Samples: {ref_samp}')

    if direction == 'up':  ##  Generating sin and cosine up chirps
        sinChirp = chirp(t_ref, fl, t_ref[-1], fh, method='linear').astype(np.float32)
        cosChirp = chirp(t_ref, fl, t_ref[-1], fh, method='linear', phi=90).astype(np.float32)

    elif direction == 'down':  ##  Generating sin and cosine down chirps
        sinChirp = chirp(t_ref, fh, t_ref[-1], fl, method='linear').astype(np.float32)
        cosChirp = chirp(t_ref, fh, t_ref[-1], fl, method='linear', phi=90).astype(np.float32)

    if window == True:  ##  Adding a window if desired
        hamming = np.hamming(ref_samp).astype(np.float32)
        sinChirp *= hamming
        cosChirp *= hamming

    print('Created Reference Signals... Now to Compress')

    az_samp = data.shape[0]
    compressed = np.zeros_like(data, dtype=np.complex64)
    chunk_size = az_samp // num_processes
    chunks = [data[i:i+chunk_size] for i in range(0, az_samp, chunk_size)]

    print('Creating chunks to compress in parallel...')
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(compress_chunk, [(chunk, sinChirp, cosChirp) for chunk in chunks])

    idx = 0
    for i in results:
        samps = i.shape[0]
        compressed[idx:idx+samps] = i
        idx += samps

    print('Range compression finished...')
    print('')

    # compressed = np.concatenate(results)

    return compressed


#######  Running Azimuth FFT's  #############

def az_fft_chunk(chunk):
    return np.fft.fft(chunk, axis=0)

def az_fft(data, num_processes=numworkers):
    az_samp, range_samp = data.shape

    chunk_size = range_samp // num_processes
    chunks = [data[:, i:i+chunk_size] for i in range(0, range_samp, chunk_size)]

    with Pool(processes=num_processes) as pool:
        fft_results = pool.map(az_fft_chunk, chunks)

    fft = np.zeros_like(data, dtype=np.complex64)

    idx = 0
    for i in fft_results:
        samps = i.shape[1]
        fft[:, idx:idx+samps] = i
        idx += samps

    return fft

def az_ifft_chunk(chunk):
    return np.fft.ifft(chunk, axis=0)

def az_ifft(data, num_processes=numworkers):
    az_samp, range_samp = data.shape

    chunk_size = range_samp // num_processes
    chunks = [data[:, i:i+chunk_size] for i in range(0, range_samp, chunk_size)]

    with Pool(processes=num_processes) as pool:
        ifft_results = pool.map(az_ifft_chunk, chunks)

    ifft = np.zeros_like(data, dtype=np.complex64)

    idx = 0
    for i in ifft_results:
        samps = i.shape[1]
        ifft[:, idx:idx+samps] = i
        idx += samps

    return ifft
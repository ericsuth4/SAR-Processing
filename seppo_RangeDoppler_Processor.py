######################################################################################
##  Eric Sutherland
##  SNOWWI Range Doppler Processor... Rewritten with a bit of parallelization
######################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from scipy.signal import chirp
from scipy.interpolate import interp1d
from simplekml import Kml
import time
import re
import pandas as pd

import functions as fun
import parallel_functions as par
import seppo_functions as sep


# %matplotlib widget

params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
          "font.family" : "serif",
          "font.serif" : ["times new roman"],
          "figure.dpi" : 150}
plt.rcParams.update(params)


def main():

    ###  Number of files and how much data to import  ###
    start_file = 260
    stop_file = 265

    start_sample = 0
    stop_sample = 40000

    ## Platform attitude basics  ##
    vp = 72  ## Platform velocity in [m/s]
    H = 2000  ## Platform height [m]
    theta_ = np.radians(45)  ## estimated look angle
    bw_el = np.radians(45)  ##  estimating elevation beamwidth
    swath_width = 10000  ## desired swath width in [m]
    L = 1  ## antenna length [m]

    ## Constants  ##
    c = 3e8
    # num_plots = 25  ## helps to determine how many lines to plot (fewer speeds things up)

    ##  radar parameters  ##
    f0 = 5.39e9
    prf = 1e3
    lambda_ = c/f0
    fs = 491.52e6*2

    ###  For NH Data  ###
    # fl = 240e6
    # fh = 320e6
    # tp = 5.7e-6
    ###  For CO Data  ###
    fl = 143e6
    fh = 223e6
    tp = 11e-6
    az_BW = 100

    chirp_direction = 'up'
    B = fh - fl


    channel = 'chan2'
    timestamp = '20240327T125534_2'


    ###############################################
    bucket = f'mirsl01-data'
    prefix = f'airborne/SNOWWI/{timestamp}/{channel}/'
    localpath = '/dev/shm'
    ###############################################
    

    ###  Before running any processing, make/check for outputs directory  ###
    outputpath = fun.mk_output_dir(localpath)
    

    filelist = sep.get_filelist_from_bucket(bucket, prefix)
    rawdata = sep.load_data_from_s3(bucket, filelist, start_idx=start_file, stop_idx=stop_file, start_sample=start_sample, stop_sample=stop_sample)

    ###  Filtering then Compressing Data  ###
    # rawdata = fun.bandpass(rawdata, fl-10e6, fh+10e6, fs, 3)
    numworkers = os.cpu_count()

    print(f'Working with {numworkers} cores...')
    print('and starting a timer...')
    print('')

    start = time.time()

    ###  Filtering then Compressing Data  ###
    rawdata = fun.bandpass(rawdata, fl-10e6, fh+10e6, fs, 3)
    compressed = par.compress(rawdata, tp, fs, fl, fh, chirp_direction, window=True, plot=False, num_processes=numworkers)
    del(rawdata)

    ### Truncate swath to be just returns  ### 
    swath = compressed[:, start_sample+19000:]
    Rmin = H
    Rmax = Rmin + ((c/fs)*swath.shape[1])/2
    del(compressed)



    ###  Finding Azimuth FFT  ###
    az_samples = swath.shape[0]
    T_az = 1/prf
    az_tmin = 0
    az_tmax = T_az*az_samples

    print('Doing azimuth FFT and range correction...')
    az_fft_data = np.fft.fftshift(par.az_fft(swath, numworkers), axes=0)
    del(swath)

    ###  Applying Range Corrections  ###
    range_corrected_rd = fun.rcmc(az_fft_data, lambda_, prf, fs, Rmin, Rmax, vp).astype(np.complex64)
    del(az_fft_data)    


    print('Checking out some Doppler...')
    dopplerpath = os.path.join(outputpath, f'{timestamp}_doppler_fit.dat')

    if channel == 'chan0' or os.path.exists(dopplerpath) == False:  ##  checking if youre processing channel 0 and there isnt currently a doppler fit file. If so, run the 
        print(f'No Doppler fit file found, creating one to be saved at {dopplerpath}')
        doppler_fit, doppler_centroids = fun.fit_doppler(range_corrected_rd, prf, fs, B, 1, False)
        fun.write_doppler_fit(doppler_fit, dopplerpath)

        plt.figure()
        plt.plot(doppler_fit)
        plt.plot(doppler_centroids)
        plt.show()


    else:
        print(f'Doppler fit found... loading from {dopplerpath}')
        doppler_fit = fun.load_doppler_fit(dopplerpath)

    ###  Creating/Applying Azimuth Matched Filter  ###
    print('Creating and applying azimuth compression...')
    az_matched_filter = fun.azimuth_compress(range_corrected_rd, lambda_, prf, az_BW, doppler_fit, Rmin, Rmax, vp).astype(np.complex64)  ##  Create matched filter
    focused_rd = (range_corrected_rd*az_matched_filter).astype(np.complex64)  ## Apply matched filter
    del(range_corrected_rd)
    del(az_matched_filter)

    print('Finally an azimuth IFFT...')
    print('')
    focused = par.az_ifft(focused_rd, numworkers)  ###  Finally do inverse fft to find focused data
    del(focused_rd)

    ##### Multilooking data to the 'theoretical radar resolution' L/2 and c/2B  #######
    focused, range_res, az_res = fun.fix_slc_resolution(focused, vp, prf, fs, L, fh, fl)

    fun.write_slc(focused, outputpath, channel, timestamp, start_file, stop_file, vp, H, f0, fs, prf, az_BW, range_res, az_res)

    end = time.time()
    print(f'Time Taken: {end-start}')

if __name__ == "__main__":
    print('Is this thing on')
    main()


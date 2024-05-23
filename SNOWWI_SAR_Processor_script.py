######################################################################################
##  Eric Sutherland
##  SNOWWI Range Doppler Processor... Rewritten in a script
##
##
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
    start_file = 200
    stop_file = 220

    start_sample = 0
    stop_sample = 70000

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
    az_BW = 160

    chirp_direction = 'up'
    B = fh - fl

    ###  Importing Files from given directory  ###
    directory = r'\\Sentinel\SNOWWI\Colorado2024\March\Radar_data\from_disk_F\save_data_nvme1n1\20240327T125534'  ### CBand CO March
    # directory = r'\\Sentinel\SNOWWI\Colorado2024\March\Radar_data\from_disk_I\save_data_nvme1n1\20240326T190019'  ### KuLow CO March

    channel = 'chan2'
    timestamp = directory.rsplit('\\', 1)[-1]

    ###  Before running any processing, make/check for outputs directory  ###
    outputpath = fun.mk_output_dir(directory)
    channelpath = os.path.join(directory, channel)

    filelist = fun.list_files(channelpath)
    rawdata = fun.load_data(filelist, start_file, stop_file, start_sample=start_sample, stop_sample=stop_sample)


    start = time.time()

    ###  Filtering then Compressing Data  ###
    rawdata = fun.bandpass(rawdata, fl-10e6, fh+10e6, fs, 3)
    compressed = fun.compress(rawdata, tp, fs, fl, fh, chirp_direction, plot=False)
    del(rawdata)

    ### Truncate swath to be just returns  ###
    swath = compressed[:, start_sample+21500:]
    Rmin = H
    Rmax = Rmin + ((c/fs)*swath.shape[1])/2
    del(compressed)

    ###  Finding Azimuth FFT  ###
    az_samples = swath.shape[0]
    T_az = 1/prf
    az_tmin = 0
    az_tmax = T_az*az_samples

    az_fft_data = np.fft.fftshift(np.fft.fft(swath, axis=0), axes=0).astype(np.complex64)
    del(swath)

    ###  Applying Range Corrections  ###
    range_corrected_rd = fun.rcmc(az_fft_data, lambda_, prf, fs, Rmin, Rmax, vp).astype(np.complex64)
    del(az_fft_data)    



    dopplerpath = os.path.join(outputpath, f'{timestamp}_doppler_fit.dat')

    if channel == 'chan0' or os.path.exists(dopplerpath) == False:  ##  checking if youre processing channel 0 and there isnt currently a doppler fit file. If so, run the 
        print(f'No Doppler fit file found, creating one to be saved at {dopplerpath}')
        doppler_fit, doppler_centroids = fun.fit_doppler(range_corrected_rd, prf, fs, B, 1, False)
        fun.write_doppler_fit(doppler_fit, dopplerpath)

        # plt.figure()
        # plt.plot(doppler_fit)
        # plt.plot(doppler_centroids)
        # plt.show()


    else:
        print(f'Doppler fit found... loading from {dopplerpath}')
        doppler_fit = fun.load_doppler_fit(dopplerpath)

    ###  Creating/Applying Azimuth Matched Filter  ###
    az_matched_filter = fun.azimuth_compress(range_corrected_rd, lambda_, prf, az_BW, doppler_fit, Rmin, Rmax, vp).astype(np.complex64)  ##  Create matched filter
    focused_rd = (range_corrected_rd*az_matched_filter).astype(np.complex64)  ## Apply matched filter

    focused = np.fft.ifft(focused_rd, axis=0).astype(np.complex64)  ###  Finally do inverse fft to find focused data

    ##### Multilooking data to the 'theoretical radar resolution' L/2 and c/2B  #######
    focused, range_res, az_res = fun.fix_slc_resolution(focused, vp, prf, fs, L, fh, fl)

    multix = int(4)
    multiy = int(multix*4 / 1.5)
    focused_multilook = fun.multilook(abs(focused), multix, multiy)  ###  Multilooking

    # ##########################  Need to write statement for whether or not this plots/saves figures  ###########################################
    plt.figure()

    plt.imshow(20*np.log10(abs(focused_multilook)), vmin=65, vmax=110, origin='upper', cmap='gray', interpolation='None')
    plt.title('CBand Grand Mesa: March 27, 2024')
    plt.xlabel('Slant Range [Samples]')
    plt.ylabel('Along Track [Samples]')
    plt.colorbar(label='[dB - Uncal]')

    image_file = f'{channel}_{timestamp}.png'
    # cwd = os.getcwd()
    plt.savefig(os.path.join(outputpath, image_file), dpi=1000, bbox_inches='tight', pad_inches=0.25, transparent=False)
    plt.close()



    fun.write_slc(focused, outputpath, channel, timestamp, start_file, stop_file, vp, H, f0, fs, prf, az_BW, range_res, az_res)

    end = time.time()
    print(f'Time Taken: {end-start}')

if __name__ == "__main__":
    print('Is this thing on')
    main()
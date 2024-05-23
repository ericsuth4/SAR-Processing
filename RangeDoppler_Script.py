# %% [markdown]
# ##  First go at a real SAR Processor
# 
# ###  Eric Sutherland
# ### 2/15/2024

# %%
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

%matplotlib inline

# %% [markdown]
# # Important constants that change with datasets #

# %%

###  Number of files and how much data to import  ###
start_file = 645
stop_file = 685
start_sample = 000
# stop_sample = 70000
stop_sample = 34000

## Platform attitude basics  ##
vp = 72  ## Platform velocity in [m/s]
H = 2000  ## Platform height [m]
theta_ = np.radians(45)  ## estimated look angle
bw_el = np.radians(45)  ##  estimating elevation beamwidth
swath_width = 10000  ## desired swath width in [m]

## Constants  ##
c = 3e8
num_plots = 25  ## helps to determine how many lines to plot (fewer speeds things up)

##  radar parameters  ##
f0 = 5.39e9
# f0 = 13.64e9
prf = 1e3
lambda_ = c/f0


## Compression parameters  ##
fs = 491.52e6*2

###  For NH Data  ###
# fl = 240e6
# fh = 320e6
# tp = 5.7e-6

###  For CO Data  ###
fl = 143e6
fh = 223e6
tp = 11e-6


chirp_direction = 'up'
B = fh - fl



# %%
# directory = r'\\seasat\Projects\SNOWWI\SNOWWI_NH\NH_Flight_December\20231215T144359'  ####  CBand in NH over airport
directory = r'\\seasat\Projects\SNOWWI\SNOWWI_NH\NH_Flight_December\20231215T142814'  ### CBand NH december
# directory = r'\\seasat\Projects\SNOWWI\Colorado_January\20240205T020242\chan0'  ####  CBand in Grand Junction
directory = r'\\Sentinel\SNOWWI\Colorado2024\March\Radar_data\from_disk_F\save_data_nvme1n1\20240327T120951\chan0'  ### CBand CO March
directory = r'\\Sentinel\SNOWWI\Colorado2024\March\Radar_data\from_disk_F\save_data_nvme1n1\20240327T124951\chan0'  ### CBand CO March
directory = r'\\Sentinel\SNOWWI\Colorado2024\March\Radar_data\from_disk_F\save_data_nvme1n1\20240327T123855\chan0'  ### CBand CO March


# directory = r'\\seasat\Projects\SNOWWI\Colorado_January\20240204T193523\chan0'  ####  Ku Low in Grand Junction

# directory = r'\\Sentinel\SNOWWI\Colorado2024\January\Radar_data\save_data_nvme0\20240205T031232\chan0'  ####  CBand in Grand Mesa


filelist = fun.list_files(directory)

rawdata = fun.load_data(filelist, start_file, stop_file, start_sample=start_sample, stop_sample=stop_sample)



# %%
timestamp = 1151496015  ###  first time stamp from folder (gps time)
unixtime = 	1702669437  ## Real (start) unix time based on timestamps

gps_to_unix = 315964800
real_gps_start = unixtime - gps_to_unix

gps_offset = real_gps_start - timestamp

file_start_timestamp = re.search(r'_(\d+)\.', filelist[start_file]).group(1)
file_stop_timestamp = re.search(r'_(\d+)\.', filelist[stop_file]).group(1)
gps_start_file = int(file_start_timestamp) + gps_offset
gps_stop_file = int(file_stop_timestamp) + gps_offset

novatel_directory = r'\\seasat\Projects\SNOWWI\SNOWWI_NH\NH_Flight_December\NovaTel_20231215'
novatel_filename = r'20231215_3.txt'

novatel = fun.readNovatel(novatel_directory, novatel_filename, gps_start_file, gps_stop_file)


# %%
# dt = 0.008  ## [s] between NovaTel samples

# dX = np.diff(novatel['X-ECEF'])
# dY = np.diff(novatel['Y-ECEF'])
# dZ = np.diff(novatel['Z-ECEF'])

# vp = np.sqrt(dX**2 + dY**2 + dZ**2)/dt

# dH = np.diff(novatel['H-Ell'])
# dHdt = dH/dt
# dHdt = np.average(dH.reshape(-1, 25), axis=1)/dt

# avg_vp = np.average(vp.reshape(-1, 25), axis=1)
# acc = np.diff(avg_vp)/dt


# %%

# rawdata = correct_height(rawdata, novatel, fs)
# print(novatel['GPSTime [sec]'])

# rawdata = np.roll(rawdata, -9100)

# %%
###  Plotting raw radar data  ###
# %matplotlib widget

n = int(rawdata.shape[0]/num_plots)

plt.figure(figsize=(16, 4))

for i in rawdata[0::n]:
    plt.plot(i)



plt.title(f'{num_plots} Raw Pulses - Sanity Check')
plt.xlabel('Range Samples')
plt.ylabel('Magnitude Samples')

plt.xlim(0, rawdata.shape[1])

plt.grid()
plt.show()

# %%


# %%


# %%
###  Compressing and plotting compression waveforms  ###

rawdata = fun.bandpass(rawdata, fl-10e6, fh+10e6, fs, 3)
compressed = fun.compress(rawdata, tp, fs, fl, fh, chirp_direction, plot=False)


del(rawdata)

# %%
# %matplotlib widget

# plt.figure()
# plt.plot(20*np.log10(abs(compressed[0])))
# plt.show()

# %%
### Truncate swath to be just returns  ###

# swath, Rmin, Rmax = truncate_swath(compressed, swath_width, H, tp, fs, theta_, bw_el)
# swath = compressed[:, start_sample+30000:]
swath = compressed[:, start_sample+21500:]

# swath = compressed[:, 8500:]


# Rmin = novatel['H-Ell'].mean()
Rmin = H
Rmax = Rmin + ((c/fs)*swath.shape[1])/2


del(compressed)



# %%
print(Rmin)
print(Rmax)

# %%
###  Plotting image of compressed data

# %matplotlib inline

# n = int(swath.shape[0]/num_plots)



# fig, ax = plt.subplots(figsize=[16, 4], nrows=1, ncols=2)

# im = ax[0].imshow(20*np.log10(abs(swath)), vmin=90, vmax=120, origin='lower', interpolation='None', cmap='gray', aspect='equal')
# ax[0].set_title('Compressed Data')
# ax[0].set_xlabel('Range Samples')
# ax[0].set_ylabel('Azimuth Samples')
# fig.colorbar(im, ax=ax[0], label='[dB]')

# for i in swath[::n]:
#     ax[1].plot(20*np.log10(abs(i)), alpha=0.3)

# ax[1].set_title('Compressed Range Lines')
# ax[1].set_xlabel('Range Samples')
# ax[1].set_ylabel('[dB]')
# ax[1].set_xlim(0, swath.shape[1])
# ax[1].grid()

# plt.show()

# %%
%matplotlib widget

plt.figure()

plt.imshow(20*np.log10(abs(swath)), vmin=90, vmax=120, origin='upper', interpolation='None', cmap='gray', aspect='equal')

image_file = f'compressed1.png'
cwd = os.getcwd()

plt.savefig(os.path.join(cwd, image_file), dpi=1000, bbox_inches='tight', pad_inches=0, transparent=True)

plt.colorbar()
plt.show()

# %% [markdown]
# ## Azimuth FFT  ##

# %%
###  Finding Azimuth FFT  ###

# swath = multilook(swath, 4, 1)

az_samples = swath.shape[0]
T_az = 1/prf
az_tmin = 0
az_tmax = T_az*az_samples

t_az = np.linspace(az_tmin, az_tmax, az_samples)
az_fft_data = np.fft.fftshift(np.fft.fft(swath, axis=0), axes=0)

az_freq = np.linspace(-prf/2, prf/2, az_samples)
freq_khz = az_freq/1e3
del(swath)


# %%
###  Plotting some Azimuth FFT Stuffs  ###

vmin = 130
vmax = 150

# vmin = 140
# vmax = 160

n = int(az_fft_data.shape[1]/num_plots)

fig, ax = plt.subplots(figsize=[16, 4], nrows=1, ncols=2)

im = ax[0].imshow(20*np.log10(abs(az_fft_data)), vmin=vmin, vmax=vmax, origin='lower', interpolation='None', extent=[0, az_fft_data.shape[1], -prf/2, prf/2], aspect='auto')
ax[0].set_title('Azimuth Frequency Spectrogram')
ax[0].set_xlabel('Range Samples')
ax[0].set_ylabel('Azimuth Frequency [Hz]')
fig.colorbar(im, ax=ax[0], label='[dB]')

for i in az_fft_data.T[::n]:
    ax[1].plot(az_freq, 20*np.log10(abs(i)), alpha=0.3)

ax[1].set_title('Azimuth Frequency of Different Range Lines')
ax[1].set_xlabel('Azimuth Frequency [Hz]')
ax[1].set_ylabel('[dB]')
ax[1].set_xlim(-prf/2, prf/2)
ax[1].grid()

plt.show()

# %% [markdown]
# ###  Applying Range Cell Migration Corrections  ###

# %%
range_corrected_rd = fun.rcmc(az_fft_data, lambda_, prf, fs, Rmin, Rmax, vp)
# range_corrected_rd = az_fft_data
del(az_fft_data)

# %% [markdown]
# ## Smoothing Doppler, finding Doppler Centroids  ##

# %%
# def fit_doppler(data, prf, fs, B, order, snr=True):


#     smoothed_doppler, range_corrected_rd = fun.smooth_doppler(data, fs, B)  ##  Running some rolling averages over azimuth doppler centroids
#     az_freq = np.linspace(-prf/2, prf/2, data.shape[0])

#     doppler_centroid_idx = np.argmax(smoothed_doppler, axis=0)

#     if snr == True:
#         means = np.mean(abs(smoothed_doppler), axis=0)
#         maxs = np.max(abs(smoothed_doppler), axis=0)

#         badsnr = np.where(maxs<(1.3*means))[0]  ## Finding indeces where snr < ...dB

#         doppler_centroid_idx[badsnr[0]:] = doppler_centroid_idx[badsnr[0] - 1]  ## replacing indeces with bad snr to the last index with a good snr
    
#     doppler_centroids = az_freq[doppler_centroid_idx]

#     range_samples = np.linspace(0, doppler_centroids.shape[0]-1, doppler_centroids.shape[0])

#     coeff = np.polyfit(range_samples, doppler_centroids, 1)
#     doppler_fit = coeff[0]*range_samples + coeff[1]  # **3 + coeff[1]*range_samples**2 + coeff[2]*range_samples + coeff[3]

#     return doppler_fit, range_corrected_rd, doppler_centroids




# %%
doppler_fit, range_corrected_rd, doppler_centroids = fun.fit_doppler(range_corrected_rd, prf, fs, B, 3, False)



# %%
plt.figure()
plt.plot(doppler_fit)
plt.plot(doppler_centroids)

plt.ylim(-500, 500)
plt.show()

# %%
# smoothed_doppler, range_corrected_rd = smooth_doppler(range_corrected_rd, fs, B)  ##  Running some rolling averages over azimuth doppler centroids

# doppler_centroid_idx = np.argmax(smoothed_doppler, axis=0)
# doppler_centroids = az_freq[doppler_centroid_idx]

# range_samples = np.linspace(0, doppler_centroids.shape[0]-1, doppler_centroids.shape[0])

# coeff = np.polyfit(range_samples, doppler_centroids, 3)
# doppler_fit = coeff[0]*range_samples**3 + coeff[1]*range_samples**2 + coeff[2]*range_samples + coeff[3]

# doppler_fit = coeff[0]*range_samples**5 + coeff[1]*range_samples**4 + coeff[2]*range_samples**3 + coeff[3]*range_samples**2 + coeff[4]*range_samples + coeff[5]
# doppler_fit = coeff[0]*range_samples**2 + coeff[1]*range_samples + coeff[2]




# %%
# def rcmc(data, lambda_, fs_az, fs_rng, Rmin, Rmax, vp):

#     az_samp = data.shape[0]
#     rng_samp = data.shape[1]

#     y = np.arange(az_samp)
#     x = np.arange(rng_samp)

#     fn = np.linspace(-fs_az/2, fs_az/2, az_samp)

#     # R0 = (Rmin + Rmax) / 2

#     R = np.linspace(Rmin, Rmax, rng_samp).reshape((-1, rng_samp))
#     fn = (np.linspace(-fs_az/2, fs_az/2, az_samp)**2).reshape((az_samp, -1))  ## squared because of range migration equation
#     Rfn = fn.dot(R)
#     dR = (lambda_**2 * Rfn) / (8*vp**2)

#     Rshift = (2*dR*fs_rng) / 3e8
#     # Rshift = x - np.array(shift_samples)[:, np.newaxis]

#     new_matrix = np.zeros_like(data)

#     # print(Rshift)
#     # return Rshift
#     xin = np.arange(rng_samp)
    
#     for i in range(az_samp):
#         x = xin - Rshift[i, :]
#         interp_func = interp1d(x, data[i], kind='cubic', bounds_error=False, fill_value=1)
#         new_matrix[i] = interp_func(np.arange(rng_samp))
    
#     return new_matrix


# %%
# rd_corrections = rcmc(az_fft_data, lambda_, prf, Rmin, Rmax, vp)  ##  this matrix is the phase corrections that need to be applied
# range_corrected_rd = az_fft_data*rd_corrections




# %%
# %matplotlib widget

# plt.figure(figsize=(9, 4))
# plt.imshow(20*np.log10(abs(range_corrected_rd)), vmin=130, vmax=150, origin='lower', cmap='viridis', interpolation='None')
# plt.colorbar()
# plt.show()

# %%
###  Plotting some Azimuth FFT Stuffs  ###

# vmin = 130
# vmax = 150

# # vmin = 140
# # vmax = 160

# n = int(az_fft_data.shape[1]/num_plots)

# fig, ax = plt.subplots(figsize=[16, 4], nrows=1, ncols=2)

# im = ax[0].imshow(20*np.log10(abs(range_corrected_rd)), vmin=vmin, vmax=vmax, origin='lower', interpolation='None', extent=[0, az_fft_data.shape[1], -prf/2, prf/2], aspect='auto')
# ax[0].plot(doppler_fit, color='k')
# ax[0].set_title('Azimuth Frequency Spectrogram')
# ax[0].set_xlabel('Range Samples')
# ax[0].set_ylabel('Azimuth Frequency [Hz]')
# ax[0].grid()
# # fig.colorbar(im, ax=ax[0], label='[dB]')

# im = ax[1].imshow(20*np.log10(abs(smoothed_doppler)), vmin=vmin, vmax=vmax, origin='lower', interpolation='None', extent=[0, az_fft_data.shape[1], -prf/2, prf/2], aspect='auto')


# ax[1].set_title('Azimuth Frequency of Different Range Lines')
# ax[1].set_xlabel('Azimuth Frequency [Hz]')
# ax[1].set_ylabel('[dB]')
# # ax[1].set_xlim(-prf/2, prf/2)
# ax[1].grid()

# plt.show()

# %%
R = np.linspace(Rmin, Rmax, range_corrected_rd.shape[1])

dR = (Rmax - Rmin)/range_corrected_rd.shape[1]
azR = dR*range_corrected_rd.shape[0]
# az = np.linspace(0, dR*range_corrected_rd.shape[0], range_corrected_rd.shape[0])


# Rmax = Rmin + ((c/fs)*range_corrected_rd.shape[1])/2



# az_matched_filter = azimuth_compress(az_fft_data, lambda_, prf, prf/2, doppler_fit, Rmin, Rmax, vp)
# doppler_zero = np.zeros_like(doppler_fit)


vp = 72
az_matched_filter = fun.azimuth_compress(range_corrected_rd, lambda_, prf, 160, doppler_fit, Rmin, Rmax, vp)




# %%

print(Rmax)
print(azR)
print(vp)

# %%
# az_matched_filter = azimuth_compress(range_corrected_rd, lambda_, prf, doppler_centroids, Rmin, Rmax, vp, 5)

# focused_rd = np.fft.fftshift(np.fft.fft(zero_doppler, axis=0), axes=0)*az_matched_filter
focused_rd = range_corrected_rd*az_matched_filter


# del(az_matched_filter)
# del(range_corrected_rd)
# focused_rd = az_fft_data*az_matched_filter

# %%
# print(az_matched_filter[1])

# %%
# plt.figure()

# plt.imshow(np.angle(range_corrected_rd), interpolation='None', cmap='hsv')

# plt.show()

# %%
plt.figure()

plt.imshow(np.angle(az_matched_filter), cmap='hsv', interpolation='None')

plt.colorbar()
plt.show()

# %%
# del(range_corrected_rd)
# del(az_matched_filter)
focused = np.fft.ifft(focused_rd, axis=0)
# focused = np.fft.ifft(range_corrected_rd, axis=0)

# del(focused_rd)
# focused = rolling_avg(focused, 62, 0)



# %%
focused_multilook = fun.multilook(abs(focused), 4, 16)


# %%
azR = azR/4

# %%
%matplotlib widget
plt.figure()

plt.imshow(20*np.log10(abs(focused_multilook)), vmin=70, vmax=130, origin='upper', cmap='gray', interpolation='None', extent=[Rmin, Rmax, azR, 0])
# plt.imshow(np.angle(focused_rd[6000:7350, 7100:7500]), origin='lower', cmap='hsv', interpolation='None')
# plt.imshow(np.angle(range_corrected_rd[6000:7350, 7100:7500]), origin='lower', cmap='hsv', interpolation='None')
plt.title('C-Band Grand Mesa: March 27, 2024')
plt.xlabel('Slant Range [m]')
plt.ylabel('Along Track [m]')
plt.colorbar(label='[dB - Uncal]')

image_file = f'CBand_GM_03272024_CR_multilook1.png'
cwd = os.getcwd()

plt.savefig(os.path.join(cwd, image_file), dpi=1000, bbox_inches='tight', pad_inches=0, transparent=True)



plt.show()

# %%
CR = focused[3000:3300, 9182]

CR_fft = 

# %%



plt.figure()

plt.plot(20*np.log10(abs(focused[3000:3300, 9182])))

plt.show()

# %%
plt.figure()

plt.plot(np.angle(focused[3000:3300, 9182]))

plt.show()

# %%
# plt.figure()

# plt.plot(np.angle(range_corrected_rd[8000:9500, 9182]))

# plt.show()

# %%
R_correct = np.linspace(Rmin, Rmax, focused_multilook.shape[1])
R_correct = np.tile(R_correct, (focused_multilook.shape[0], 1))

R_correct = R_correct**2

print(R_correct.shape)

# %%


# %%
# plt.figure()

# plt.imshow(20*np.log10(abs(focused_multilook * R_correct)), vmin=200, vmax=260, origin='lower', cmap='gray', interpolation='None')
# # plt.imshow(np.angle(focused_rd[6000:7350, 7100:7500]), origin='lower', cmap='hsv', interpolation='None')
# # plt.imshow(np.angle(range_corrected_rd[6000:7350, 7100:7500]), origin='lower', cmap='hsv', interpolation='None')
# plt.title('Poorly Focused')
# plt.xlabel('Range [Samples]')
# plt.ylabel('Azimuth [Samples]')
# plt.colorbar(label='[dB - Uncal]')

# # image_file = f'CBand_windowedFocused_singlelook.png'
# # cwd = os.getcwd()

# # plt.savefig(os.path.join(cwd, image_file), dpi=1000, bbox_inches='tight', pad_inches=0, transparent=True)



# plt.show()

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%




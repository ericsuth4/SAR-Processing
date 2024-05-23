import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from skspatial.objects import Line, Points
from skspatial.plotting import plot_3d

######################################   Functions for NovAtel Data  #############################################

def get_header(file):
    return np.fromfile(file, dtype=np.uint16, count=48)

def timestamp_from_header(header):
    left = 4*7
    right = left + 4
    radio_time = header[left:right]
    vec = np.array([2**16, 2**0, 2**48, 2**32])
    sum = np.dot(radio_time.astype(np.uint64), vec.astype(np.uint64))
    return sum/122.88e6/4

def readNovatel(novatel_directory, novatel_filename, start_time, stop_time):  ###  returns the attitude data from flightline... specified by gps timestamps  ###
    novatel_file = os.path.join(novatel_directory, novatel_filename)

    cols = ['GPSTime [HMS]', 'Date', 'Week', 'GPSTime [sec]', 'Latitude', 'Longitude', 'H-Ell', 'H-MSL', 'Undulation', 'X-ECEF', 'Y-ECEF', 'Z-ECEF', 'Pitch', 'Roll', 'Heading', 'COG']
    novatel = pd.read_csv(novatel_file, delim_whitespace=True, skiprows=18, skipfooter=3, on_bad_lines='warn', names=cols, engine='python')

    gps_start = start_time - (novatel['Week'][0]*604800)  ## choosing some arbitrary start/stop time for testing code
    gps_stop = stop_time - (novatel['Week'][0]*604800)

    flightline = novatel[(novatel['GPSTime [sec]'] >= gps_start) & (novatel['GPSTime [sec]'] <= gps_stop)]  ## truncates data to only take the desired flightline
    flightline.reset_index(inplace=True)

    return flightline

def get_time(novatel):  ###  finds sampling period from novatel data
    gpstime = novatel['GPSTime [sec]'].to_numpy()
    dt = np.diff(novatel['GPSTime [sec]'])[0]
    return gpstime, dt

def get_yaw_pitch_roll(novatel):

    yaw = (novatel['Heading'] - novatel['COG']).to_numpy()
    pitch = novatel['Pitch'].to_numpy()
    roll = novatel['Roll'].to_numpy()
    yaw[yaw<-180] += 360  ## so that phase doesnt wrap

    print(f'Mean Yaw:   {yaw.mean()}')
    print(f'Mean Pitch: {pitch.mean()}')
    print(f'Mean Roll:  {roll.mean()}')

    return yaw, pitch, roll

def get_xyz(novatel):

    x = novatel['X-ECEF'].to_numpy()
    y = novatel['Y-ECEF'].to_numpy()
    z = novatel['Z-ECEF'].to_numpy()

    return x, y, z

def interpolate_attitude(yaw, pitch, roll, time, dt, samps, kind='linear'):

    # oversample = int(dt/desired_dt)
    # N = len(yaw)*oversample

    newtime = np.linspace(time[0], time[-1], samps+1)

    yaw_interp = interp1d(time, yaw, kind=kind)
    pitch_interp = interp1d(time, pitch, kind=kind)
    roll_interp = interp1d(time, roll, kind=kind)

    yaw = yaw_interp(newtime)
    pitch = pitch_interp(newtime)
    roll = roll_interp(newtime)

    return yaw, pitch, roll, newtime


def interpolate_xyz(x, y, z, time, dt, samps, kind='linear'):

    # oversample = int(dt/desired_dt)
    # N = len(x)*oversample
    

    newtime = np.linspace(time[0], time[-1], samps+1)

    x_interp = interp1d(time, x, kind=kind)
    y_interp = interp1d(time, y, kind=kind)
    z_interp = interp1d(time, z, kind=kind)

    x = x_interp(newtime)
    y = y_interp(newtime)
    z = z_interp(newtime)

    return x, y, z, newtime

def fit_position(x, y, z):

    points = np.column_stack((x, y, z))
    line_fit = Line.best_fit(points)

    projection = np.zeros_like(points)

    for i in range(points.shape[0]):
        projection[i] = line_fit.project_point(points[i])

    return projection[:, 0], projection[:, 1], projection[:, 2]
import os
import glob
import boto3
import struct
import numpy as np

def get_filelist_from_bucket(bucket, prefix):
    s3 = boto3.client('s3')
    result = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')

    file_keys = []
    for i in result['Contents']:   ###  ['Contents'] gives us a list of dictionaries
        file_keys.append(i['Key'])   ## Each dictionary corresponds to a file in the folder, where ['Key'] is specifically the file name from this dict

    file_keys = sorted(file_keys, key=lambda x: float(x.split('_')[-1][:-4]))  ## Sorting Files

    print(f'Printing file names {prefix}...')
    print('-------------------------------------------------------------------------')
    for f in file_keys:
        print(f)

    return file_keys

def download_from_s3(bucket, key, local_path):
    s3 = boto3.client('s3')
    with open(local_path, 'wb') as f:
        s3.download_fileobj(bucket, key, f)

def s3_data_to_array(bucket, key, localdir='/dev/shm'):
    filename = key.split('/')[-1]
    localpath = os.path.join(localdir, filename)

    download_from_s3(bucket, key, localpath)
    temp_data = np.fromfile(localpath, dtype=np.int16)
    os.remove(localpath)

    return temp_data

def load_data_from_s3(bucket, file_list, start_idx=0, stop_idx=-1, start_sample=0, stop_sample=0):

    ###  Given a list of file paths to .dat files of binary radar data
    ###  Looks at date code, determines how many samples should belong in each window
    ###  Returns a single matrix that holds all desired data

    first_path = file_list[0].split('/')[-3]
    pattern = r'(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})'  ## using this pattern to find a timestamp

    year = int(re.search(pattern, first_path).group(1)) 

    if year < 2024:
        n_samp = 60000

    else:
        n_samp = 64000

    file_data = []
    idx = 0
    total_pulses = 0

    for f in file_list[start_idx:stop_idx]:  ## iterate through files of given indeces
        temp_data = s3_data_to_array(bucket, f)
        temp_data = temp_data.reshape(-1, n_samp)

        print(f'File Index: {idx} with shape {temp_data.shape}')
        file_data.append(temp_data)  ##  adding data to a list. file_data becomes a list of 2D matrices, each item corresponds to a single .dat file

        idx +=1
        total_pulses+= temp_data.shape[0]  ##  this is counting number of pulses in each file

    full_data = np.zeros((total_pulses, n_samp))  ## Allocating memory for the full data array

    idx = 0   ## resetting counter

    for data in file_data:  ## iterate through items (2D matrices) adding them to specific indeces of full data matrix
        n_pulses = data.shape[0]
        full_data[idx:idx+n_pulses] = data  ## indexing where each data chunk belongs
        del(data)
        idx += n_pulses

    del(file_data)
    del(idx)
    del(total_pulses)

    if start_sample or stop_sample != 0:
        full_data = full_data[:, start_sample:stop_sample]

    print('')
    print(f'Full Data Shape: {full_data.shape}')
    print(f'First Filename: {file_list[start_idx]}')
    print(f'Last Filename: {file_list[stop_idx]}')
    print('')

    return full_data/(2**4) ###  divide by 16 because it is 12 bit data loaded as 16 bit data
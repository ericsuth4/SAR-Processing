import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import functions as fun

params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
          "font.family" : "serif",
          "font.serif" : ["times new roman"],
          "figure.dpi" : 150}
plt.rcParams.update(params)

def main():

    N = 64000

    directory = sys.argv[0]
    numfiles = int(sys.argv[1])
    # print(directory)

    goodlines, badlines = fun.checkdir(directory, numfiles)

    print(f'Finding runs with more than {numfiles} files....')
    print('')

    for line in goodlines:

        print(f'Working in following directory: {line}')
        print('')

        # outputpath = fun.mk_output_dir(line)
        outputpath = line
        

        for channum in [0, 1, 2, 3]:

            print(f'Starting Channel {channum}')

            chan = f'chan{channum}'
            indirectory = os.path.join(line, chan)  ###  Looks in the directory + channel name

            filelist = fun.list_files(indirectory, pnt=False)  ###  creates list of all .dat files for this channel
            filenum = len(filelist)

            print('Files are listed and sorted...')

            rawdata = np.zeros((filenum, N))  ### allocating memory

            for i in range(filenum):  ###  Read first window of each file
                rawdata[i, :] = fun.read_first(filelist[i], N)

            print('Data is loaded, now to plot...')

            plt.figure()
            plt.imshow(20*np.log10(abs(rawdata)+1), vmin=30, vmax=70, cmap='gray', aspect='auto', interpolation='None')  ### add 1 to get rid of log(0)
            plt.title(f'First Window of Each File - {chan}')
            plt.xlabel('Range Samples')
            plt.ylabel('Azimuth Samples [File Number]')
            plt.colorbar(label='[dB - Uncal]')

            outname = f'{chan}_Quick_look.png'
            outputfile = os.path.join(outputpath, outname)

            plt.savefig(outputfile, dpi=500, bbox_inches='tight', pad_inches=0.5)
            plt.close()

            print(f'Saving File at: {outputfile}')
            print(f'Finished Channel {channum}')
            print('---------------------------')
            print('')

    

    if len(goodlines) >= 1:
        print(f'Lines with more than {numfiles} files are:')
        for i in goodlines:
            print(i)    


    else:
        print(f'Either no lines exist with more than {numfiles} files... ')  
        print('Or they already have Quick Looks available')


    if len(badlines) >= 1:
        print(f'Lines with less than {numfiles} files are:')
        for i in badlines:
            print(i) 

    else:
        print(f'Either no lines exist with less than {numfiles} files... ')  
        print('Or they already have Quick Looks available')
    
    print('')



if __name__ == "__main__":

    sys.argv = sys.argv[1:]
    print('')

    main()

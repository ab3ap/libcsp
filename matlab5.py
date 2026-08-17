'''
This file provides two subroutines, one to read and one to write a Matlab 5.0
file.

Mike Markowski
mike.ab3ap@gmail.com
Mar 2026
'''

import numpy as np
import scipy.io as sio

def mat5Read(matlabFile):
    '''After the Matlab 5.0 file is read, the signal data and related
    information are returned.

    Input:
      matlabFile (string): file name of Matlab file.
    Output:
      iq (complex[]): the complex IQ signal data.
      fs_Hz (float): Hz, sample rate of signal.
      fc_Hz (float): Hz, center frequency of signal.
      bw_Hz (float): Hz, bandwidth of signal.
    '''

    bw_Hz = fc_Hz = None
    hdr = sio.loadmat(matlabFile) # Dictionary of header var/vals.
    for var in hdr.keys():
        if var == 'FreqValidMax':
            f1_Hz = hdr[var][0][0]
        elif var == 'FreqValidMin':
            f0_Hz = hdr[var][0][0]
        elif var == 'InputCenter':
            fc_Hz = hdr[var][0][0]
        elif var == 'Span':
            bw_Hz = hdr[var][0][0]
        elif var == 'XDelta':
            fs_Hz = 1/hdr[var][0][0]
        elif var == 'Y':
            iq = hdr[var].flatten() # I/Q data of signal.
    if bw_Hz is None:
        bw_Hz = f1_Hz - f0_Hz
    if fc_Hz is None:
        fc_Hz = (f0_Hz + f1_Hz)/2
    return iq, fs_Hz, fc_Hz, bw_Hz

def mat5Write(fname, iq, fc_Hz, bw_Hz, fs_Hz):
    '''Using data passed, the subroutine will create a Matlab 5.0 file
    containing the IQ signal.

    Input:
      fname (string): name of file to create.
      iq (complex[]): complex IQ signal samples.
      fc_Hz (float): Hz, center frequency of signal.
      bw_Hz (float): Hz, bandwidth of signal.
      fc_Hz (float): Hz, sample rate.
    Output:
      None, file created.
    '''

    # Create header dictionary.
    d = {}
    d['FreqValidMax'] = np.array([[fc_Hz + fs_Hz/2]])
    d['FreqValidMin'] = np.array([[fc_Hz - fs_Hz/2]])
    d['InputCenter'] = np.array([[fc_Hz]])
    if not bw_Hz is None:
        d['Span'] = np.array([[bw_Hz]])
    d['XDelta'] = np.array([[1/fs_Hz]])
    d['XUnit'] = np.array(['s'])
    d['YUnit'] = np.array(['V'])
    y = np.reshape(iq, (-1, 1))
    d['Y'] = y

    if fname.find('.mat') == -1:
        fname += '.mat' # Tack on .mat if user forgot it.
    sio.savemat(fname, d)


'''
Mike Markowski
mike.ab3ap@gmail.com
Jun 2026
'''

import numpy as np

def dBW_to_W(val_dB):
    return 10**(val_dB/10)

def engNot(x, sigfigs=0):
    '''Return x in engineering notation.  That is, mantissa and exponent
    where mantissa is between -999 and 999, and exponent a multiple of
    3.

    Inputs:
      x (float): number to represented in engineering notation.
      sigFigs (int): number of significant figures in output.
    Outputs:
      quantity (float): number 
    '''

    exp = int(np.floor(np.log10(abs(x))))
    exp3 = (exp//3)*3 # Make multiple of 3, e.g., 7 -> 6.
    mant = x/10**exp3

    dp = sigfigs - int(np.floor(np.log10(abs(mant)))) - 1 # Dec places.
    dp = max(0, dp)
    fmt = '%%.%df' % dp

    mantDigits = int(np.floor(np.log10(abs(mant))))
    mant = round(mant, sigfigs - mantDigits - 1) # Round to sigfigs.

    dp = max(0, sigfigs - mantDigits - 1)

    return mant,exp3,fmt

def mapRange(from0, from1, to0, to1, valFrom):
    '''Map a value from an input range to an output range.  Plain old 2D
    linear interpolation.

    Inputs:
      from0, from1 (float): start, stop of input range.
      to0, to1 (float): start, stop of output range.
      valFrom (float): value to be mapped.
    '''

    frac = (valFrom - from0)/(from1 - from0) # Normalize to 'from' range.
    vTo = to0 + frac*(to1 - to0) # Un-normalize to 'to' range.
    return vTo

def power_W(sig_V):
    return np.mean(np.abs(sig_V)**2)

def psd(iq, fs_Hz=1, pulseWidth=0.01):
    '''Calculate power spectral density of a complex signal.

    Inputs:
      iq (complex[]): the signal.
      fs_Hz (float): Hz, sample rate.  Used to scale PSD.
    '''

    psd = np.abs(np.fft.fft(iq))**2
    psd /= fs_Hz*iq.size # Scale to satisfy Parseval.
    psd = np.fft.fftshift(psd) # DC to center.
    w = int(pulseWidth*iq.size) # Sa, smoothing window width.
    if w >= 1: # Smooth psd.
        sWin = np.ones(w)/w
        psd = np.convolve(psd, sWin, mode='same')
    psd[psd==0] = 1e-200 
    psd_db = 10*np.log10(psd)
    return psd_db

def rms(sig_V):
    # return np.sqrt(power_W(sig_V))
    return np.sqrt(np.mean(np.abs(sig_V)**2))

def siNum(numUnits):
    '''Input assumed to be in format "number si" where 'si' is a standard SI
    prefix.  For example, string '4 k' or '4k' yield 4000 as return value.
    '''

    try:
        return int(numUnits) # Maybe it's an int.
    except ValueError:
        pass
    try:
        return float(numUnits) # Maybe it's a float.
    except ValueError:
        pass

    pwr={'p':-12,'n':-9,'u':-6,'m':-3,'':0,'k':3,'M':6,'G':9,'T':12}

    s = numUnits.strip()
    mag = float(s[:-1])
    units = s[-1] # Last char is SI letter prefix.
    try:
        scale = 10**pwr[units]
    except KeyError:
        print('siNum: unknown SI prefix \'%s\'' % units)
        return np.nan
    return mag*scale

def siPrefix(exp, space=False):
    '''Provide an SI prefix for an exponent.  The exponent is assumed to be
    a multiple of 3.  A space can be provided for formatting convenience.
    '''
    pre={-12:'p',-9:'n',-6:'u',-3:'m',0:'',3:'k',6:'M',9:'G',12:'T'}
    sp = ' ' if space else ''
    return ('e%d' % exp) if abs(exp)>12 else (sp + pre[exp])

def W_to_dBW(val_W):
    return 10*np.log10(val_W)

def zeroPad(x, n):
    '''Zero pad x until its length is an even multiple of n.

    Inputs:
      x (numpy.array): signal to be zero padded.
      n (int): x will be zero padded to be a multiple of n.
    Output:
      (numpy array): copy of x, zero padded as necessary.
    '''

    rem = x.size%n
    if rem == 0:
        return x.copy()
    pad = n - rem
    xz = np.zeros(x.size+pad, dtype=x.dtype)
    xz[:x.size] = x
    return xz


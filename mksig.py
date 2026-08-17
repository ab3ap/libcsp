#!/usr/bin/env python

# This code implements mathematics presented by Chad Spooner in
# his blog at https://cyclostationary.blog/ to create signals for
# CSP analysis.
#
# Subroutines are in alphabetical order.
#
# Variable names use the format thing_units, where underscore separates name
# from its units.  E.g., fc_Hz is carrier freq in Hz.
#
# mkSig() is the method of main interest.  Nearly all other methods support
# mkSig().
#
# Mike Markowski
# mike.ab3ap@gmail.com
# Mar 2021 original
# Jun 2026

# To recreate signals for Chad's cyclostationary.blog:
#
#   mkSig.py -t bpsk -s 10 -n 4000 -c 0.05 -b 0.1
#
# meaning BPSK with S/N 10dB, 4000 symbols, fc=0.05 Hz and baud=0.1.

from numpy import cos, log10, pi, sin, sqrt
import matlab5 as ml
import matplotlib.pyplot as plt
import numpy as np
import os, sys

def bpskDual(writeFile=False):
    '''
    Generate two BPSK signals with slightly different center frequencies and
    different symbol rates.  This provides a nice example to show off
    CSP's abilities.  Values from example in Eric April's 1991 paper, "The
    Advantages of Cyclostationary Processing."
    '''

    c = np.array([-1, 1])    # BPSK I-only constellation.
    factor = 2               # BPSK2 has double symbol rate of BPSK1.
    T1_bit = 16              # Sa/sym in BPSK1.
    T2_bit = factor*T1_bit   # Sa/sym in BPSK2.
    nSym1 = 4000             # Syms in generated BPSK1 signal.
    nSym2 = nSym1 // factor  # Syms in generated BPSK2 signal.
    fc1 = 3.3/32             # 3.3 Hz normalized.
    fc2 = 4.0/32             # 4.0 Hz normalized.
    ind1 = np.random.randint(0, 2, nSym1) # Random symbol sequence.
    ind2 = np.random.randint(0, 2, nSym2)
    syms1 = np.repeat(c[ind1], T1_bit) # Baseband BPSK signal.
    syms2 = np.repeat(c[ind2], T2_bit)
    bpsk1 = mix(syms1, fc1) # Mix to fc.
    bpsk2 = mix(syms2, fc2)
    sig = mkSnr(bpsk1+bpsk2, snr_dB=10)
    if writeFile:
        fc_Hz = 0.5
        bw_Hz = fs_Hz = 1
        writeSig(sig, 'bpskDual', fc_Hz, bw_Hz, fs_Hz)
    return sig

def cli(argv):
    global prog

    c = configDefault() # Start with defaults.
    prog = os.path.basename(argv[0])
    argv = argv[1:]
    i = 0
    while i < len(argv):
        opt = argv[i]
        if opt == '-b': # Baud, symbol rate.
            i += 1
            arg = argv[i]
            c['baud'] = typeVerify(arg, float)
        elif opt == '-c': # Center freq in Hz.
            i += 1
            arg = argv[i]
            c['fc_Hz'] = typeVerify(arg, float)
        elif opt == '-d': # Dual BPSK example.
            bpskDual(writeFile=True)
            sys.exit(0)
        elif opt == '-f': # Use square root raised cosine pulse shaping filter.
            c['srrc'] = True
        elif opt == '-h': # Help.
            usage()
        elif opt == '-n': # Number of symbols.
            i += 1
            arg = argv[i]
            c['nSym'] = typeVerify(arg, int)
        elif opt == '-o': # Output file.
            i += 1
            arg = argv[i]
            c['out'] = arg
        elif opt == '-p': # Phase noise intensity.
            i += 1
            arg = argv[i]
            c['Lintense'] = typeVerify(arg, float)
        elif opt == '-r': # Sample rate in Hz.
            i += 1
            arg = argv[i]
            c['fs_Hz'] = typeVerify(arg, float)
        elif opt == '-s': # Sig to noise in dB.
            i += 1
            arg = argv[i]
            c['snr_dB'] = typeVerify(arg, float)
        elif opt == '-t': # Signal type.
            i += 1
            arg = argv[i]
            c['sig'] = arg.strip().lower()
        else:
            usage()
        i += 1
    if c['sig'] not in ['bpsk', 'qpsk', '4psk', '8psk',
        '4qam', '16qam', '64qam']:
        msg = 'Missing -t {16qam|4psk|4qam|64qam|8psk|bpsk|qpsk} '
        msg += 'signal type.\n'
        print(msg)
        usage()

    return c

def configDefault():
    '''Return config with default values.
    '''

    # Defaults.
    baud = 0.1       # 0.1 baud.
    fc_Hz = 0        # Centered at 0 Hz.
    fs_Hz = 1        # 1 Hz = 1 Sa/sec sample rate.
    Lintense = 0     # Phase noise intensity.
    nSym = 10        # symbols.
    out = ''         # Output file name.
    sigType = 'bpsk' # Signal type.
    snr_dB = 30      # dB.
    sps = 1          # Samples per symbol.
    srrc = False     # Use square root raised cosine pulse shaping.

    # Create configuration dictionary as return value.
    config = {}
    config['baud'] = baud
    config['fc_Hz'] = fc_Hz
    config['fs_Hz'] = fs_Hz
    config['Lintense'] = Lintense
    config['nSym'] = nSym
    config['out'] = out
    config['sig'] = sigType
    config['snr_dB'] = snr_dB
    config['sps'] = sps
    config['srrc'] = srrc
    return config

def constellationPsk(m):
    return np.exp(2j*pi*np.arange(m)/m) # m-PSK constellation.

def constellationQam(dots):
    '''
    Input:
      dots (int): make dots x dots constellation.
    '''
    ind = np.arange(-(dots-1), dots, 2)/(dots-1) # 'dots' odds about 0.
    const = np.zeros((dots, dots), dtype=complex)
    for i,I in enumerate(ind):
        for j,Q in enumerate(ind):
            const[i,j] = I + 1j*Q
    return const

def demodBpsk(sig, fc_Hz, sps=1):
    '''Naive BPSK demodulator.

    Inputs
      sps (int) : samples/symbol.
    '''
    bb = mix(sig, -fc_Hz) # Baseband.
    if sps > 1: # Replace all samples in symbol with their average.
        for sym in np.arange(sig.size, dtype=int) // sps: # Symbols in signal.
            i = sym*sps # Index of first sample in symbol number 'sym'.
            bb[i:i+sps] = np.mean(bb[i:i+sps])
    bb[bb<0] = -1 # Smooth 0s.
    bb[bb>0] =  1 # Smooth 1s.
    return bb

def mix(sig_iq, fc_Hz, dt_s=1, complexMix=True):
    '''Mix a given signal to a carrier frequency.

    Inputs:
      sig_iq (complex[]) : Signal to mix to some carrier.
      fc_Hz (float) : Carrier frequency in Hz to mix signal to.
      dt_s (int) : <= 1/sample rate.  Default is 1 for normalized frequency.

    Output:
      (complex[]) : sig_iq mixed to carrier fc_Hz.
    '''

    n = sig_iq.size
    t_s = np.linspace(dt_s, n*dt_s, n)
    if complexMix: # Complex signal.
        carrier_iq = np.exp(2j*pi*fc_Hz*t_s)
    else: # Real signal (for use with older literature).
        carrier_iq = sin(2*pi*fc_Hz*t_s)
    return sig_iq*carrier_iq

def mkSig(config):
    '''Make a signal using the passed dictionary values.

    Input:
      config (dictionary) : See configDefault() for description of entries.
    Output:
      (complex) : i/q samples making up signal.
    '''

    baud     = config['baud']   # Data rata in bits/sec.
    fc_Hz    = config['fc_Hz']  # Center frequency in Hz.
    fs_Hz    = config['fs_Hz']  # Sample rate in Sa/sec.
    Lintense = config['Lintense'] # Phase noise intensity.  XXX Poorly done!
    nSym     = config['nSym']   # Number of symbols in generated signal.
    outfile  = config['out']    # File to create, ignored here.
    sigType  = config['sig']    # Modulation to use.
    snr_dB   = config['snr_dB'] # S/N ratio of generated signal.
    useSrrc  = config['srrc']   # Use SRRC pulse shaping.

    symPerMod = {'bpsk':2, '4psk':4, 'qpsk':4, '8psk':8,
        '4qam':4, '16qam':16, '64qam':64}
    m = symPerMod[sigType] # Number of symbols in modulation scheme.

    # Create baseband i/q signal with 1 sample/symbol.
    if sigType[-3:] == 'psk':
        ind = np.random.randint(0, m, nSym) # Random symbol sequence.
        c = constellationPsk(m) # m-PSK constellation.
        sig = c[ind]
    else: # 'qam'
        dots = int(sqrt(m))
        indI = np.random.randint(0, dots, nSym) # Random symbol sequence.
        indQ = np.random.randint(0, dots, nSym) # Random symbol sequence.
        c = constellationQam(dots)
        sig = c[indI, indQ]

    # Calculate sps, samples/symbol.
    sps = int(fs_Hz/baud) # (Sa/s)/(sym/s) = Sa/sym
    config['sps'] = sps

    if not useSrrc: # No pulse shaping.
        sig = np.repeat(sig, sps) # sps Sa/sym.
    else: # Use square root raised cosine pulse shaping.
        h = srrc(sps)
#       h *= sps # XXX For testing, restore unit sym gain from SRRC's 1/Ts.
        z = np.zeros(sig.size*sps, dtype=complex)
        z[::sps] = sig # Each bit followed by sps-1 zeros.
        sig = np.convolve(z, h, mode='full') # Pulse shape.
        delay = int(len(h)//2)
        sig = sig[delay:delay+len(z)]

    sigTx = mix(sig, fc_Hz, 1/fs_Hz) # Mix to carrier frequency.
    sigRx = mkSnr(sigTx, snr_dB) # Received sig across AWGN channel.
    # Phase noise is dBc/Hz at given offsets.  XXX Here, it is random. Hack!
    if Lintense > 0:
        L = np.exp(1j*Lintense*np.random.randn(sigRx.size)) # Phase noise.
        sigRx *= L
    return sigRx

def mkSnr(s_V, snr_dB):
    '''Add noise to a noise-free signal to achieve specified signal to
    noise ratio.

    Inputs:
      s_V (complex[]) - input signal
      snr_dB (float) - desired signal to noise ratio in units of dB.

    Output:
      (complex[]) - noisy version of input signal.
    '''

    if len(s_V) == 0:
        return s_V

    # Calculate noise power level needed to reach desired s/n.
    snrDesired = 10**(snr_dB/10) # Linear ratio.
    s_W = power_W(s_V)
    nDesired_W = s_W/snrDesired

    # Create noise at desired level.
    n_V = (np.random.randn(s_V.size) + 1j*np.random.randn(s_V.size))
    n_W = power_W(n_V)
    n_V *= sqrt(nDesired_W/n_W)

    return s_V + n_V

def iqPlot(sig, config):
    '''Simple routine to plot i/q signal.
    '''

    if config['srrc']:
        print('IQ plot not provided with pulse shaping because')
        print('matched filter and clock recovery are needed.')
        return
    if sig is None or len(sig) == 0:
        print('No signal to plot.')
        return
    plt.title(config['sig'].upper())
    plt.xlabel('In-phase')
    plt.ylabel('Quadrature')
    if np.max(np.imag(sig)) < 0.8:
        plt.ylim(-1, 1)
    plt.plot(np.real(sig), np.imag(sig), '.', ms=3)
    plt.grid(True)
    plt.show()

def power_W(sig_V):
    return np.mean(np.abs(sig_V)**2)

def srrc(Ts, beta=0.35, nTaps=100):
    '''Construct a square root raised cosine filter.
    Impulse response, h(t), from
    https://en.wikipedia.org/wiki/Root-raised-cosine_filter

    Inputs:
      Ts (float) : reciprocal of symbol rate, or samples/symbol.
      nTaps (int) : number of taps in filter.
      beta (float) : roll off factor.
    '''

    t = np.arange(-nTaps//2, nTaps//2+1)

    # General case.
    num = 4*beta*t/Ts
    num *= cos(pi*t/Ts*(1 + beta))
    num += sin(pi*t/Ts*(1 - beta))
    denom = pi*t*(1 - (4*beta*t/Ts)**2)
    denom[denom==0] = 1 # Singularities handled below.
    h = num/denom

    # Singularity t = 0 case.
    z = np.where(t == 0)
    h[z] =  1/Ts*(1 + beta*(4/pi - 1))

    # Singularity t = +/- Ts/4Beta cases.
    z1 = np.where(t == -Ts/(4*beta))
    z2 = np.where(t ==  Ts/(4*beta))
    if len(z1[0]) > 0 or len(z2[0]) > 0:
        v =  (1 + 2/pi)*sin(pi/(4*beta))
        v += (1 - 2/pi)*cos(pi/(4*beta))
        v *= beta/(Ts*sqrt(2))
        h[z1] = h[z2] = v

    return h

def typeVerify(val, valType, errMsg=None):
    '''Verify a value is specified type, and print if it isn't.  For example,

    typeVerify('12.3', float, 'Oh no')
    12.3

    typeVerify('12.3', int, 'Oh no')
    Oh no

    Input:
      val : value to be type checked.
      valType : type to check 'val' against.
    Output:
      valType or exit.
    '''

    global prog

    try:
        return valType(str(val))
    except ValueError:
        if errMsg == None:
            errMsg = '%s not %s.\n' % (str(val), str(valType))
        print(errMsg)
        usage()

def usage():
    global prog

    print('Usage: %s [-b baud] [-c fc] [-d] [-f] [-h] [-n n] [-o out] ' % prog)
    print('  [-r fs] [-s snr] [-t type]')
    print()
    print('-b: default 0.1, baud or symbols/sec.')
    print('-c: default 0, center freq in Hz.')
    print('-d: dual BPSK example, creates bpskDual.iq and bpskDual.mat.')
    print('-f: use square root raised cosine pulse shaping filter.')
    print('-h: help message.')
    print('-n: default 10, number of symbols.')
    print('-o: output file name without extension.')
    print('-p: default 0, phase noise random intensity.')
    print('-r: default 1, sample rate in Hz.')
    print('-s: default 30, SNR in dB.')
    print('-t: default bpsk, type of signal.  Any of')
    print('    bpsk, qpsk, 4psk, 8psk, 4qam, 16qam, 64qam.')

    sys.exit(1)

def writeSig(iq, filename, fc_Hz, bw_Hz, fs_Hz):
    with open(filename+'.iq', 'wb') as f:
        iq.astype(np.complex64).tofile(f)

    ml.mat5Write(filename+'.mat', iq, fc_Hz, bw_Hz, fs_Hz)

#
#   m a i n
#

def main(argv):

    c = cli(argv)      # Dictionary of settings.
    outfile = c['out'] # Matlab file to create.
    fc_Hz = c['fc_Hz'] # Center frequency in Hz.
    fs_Hz = c['fs_Hz'] # Sample rate in Hz.
    sigType = c['sig'] # Modulation.

    sig = mkSig(c)
    if len(sig) == 0:
        print('Zero length signal.  Quitting.')
        sys.exit(1)
    bw_Hz = c['fs_Hz']
    writeSig(sig, outfile, fc_Hz, bw_Hz, fs_Hz)
    print('%s .iq and .mat created.' % outfile)

    sig = mix(sig, -fc_Hz, 1/fs_Hz)
    plt.title('Bit Stream')
    plt.plot(sig.real, '.-', lw=0.75, ms=3)
    plt.plot(sig.imag, '.-', lw=0.75, ms=3)
    plt.grid(True)
    plt.show()

    iqPlot(sig, c) # I/Q plot.

if __name__ == '__main__':
    main(sys.argv)

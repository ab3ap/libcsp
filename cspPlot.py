# Routines for plotting CSP results.  A bit hackish!

# Mike Markowski
# mike.ab3ap@gmail.com
# Oct 2021 original
# Jun 2026

from matplotlib.collections import PolyCollection
from numpy import log10
import libcsp as csp
import matplotlib.pyplot as plt
import numpy as np
import util

def cycles(alphas, scfs, fc_Hz=0, fs_Hz=1, fout=None):

    s = []
    for i in range(alphas.size):
        s.append(max(np.abs(scfs[i])))
    s = np.array(s)
    _, yPwr10, _ = util.engNot(max(s))
    yPre = util.siPrefix(yPwr10)
    s /= 10**yPwr10

    _, xPwr10, _ = util.engNot(fs_Hz)
    xPre = util.siPrefix(xPwr10)
    a = (fs_Hz*alphas)/10**xPwr10

    fig, ax = plt.subplots()
    ax.set_box_aspect(1/2) # Graph height is 1/2 x width.
    plt.grid(True)
    plt.xlabel('Cycle Freq (%sHz)' % xPre)
    if yPwr10 == 0:
        plt.ylabel('SCF Magnitude (linear)')
    else:
        plt.ylabel(r'SCF Magnitude (x$10^%d$)' % yPwr10)

    a0 = min(a)
    a1 = max(a)
    gap = 0.02*(a1 - a0)
    plt.xlim(a0-gap, a1+gap)

    ymin = min(s)
    ymax = max(s)
    gap = 0.1*(ymax - ymin)
    plt.ylim(0, ymax + gap)

    plt.vlines(a, 0, s, linewidth=0.5) # Vertical bars.
    plt.scatter(a, s, marker='x')      # Peak markers.
    if fout == None:
        plt.show()
    else:
        plt.savefig(fout, bbox_inches='tight', format='png')

def scatter(quads, key=None, fc_Hz=0, fs_Hz=1, fout=None):

    _, pwr10, _ = util.engNot(fs_Hz) # Find if kHz, MHz, GHz, etc.
    pre = util.siPrefix(pwr10)       # Convert power of 10 to SI char prefix.
    fig, ax = plt.subplots()
    plt.grid(True)
    ax.set_box_aspect(3/4) # Graph height is 1/2 x width.

    #
    #   E x t r a c t   D a t a
    #
    freq  = (fs_Hz*quads.transpose()[0])/10**pwr10
    alpha = fs_Hz*quads.transpose()[1]/10**pwr10

    #
    #   S c a t t e r   P l o t
    #
    # plt.scatter(freq, alpha, s=5)
    magnitude = np.log10(quads.transpose()[2]) # log(SCF)
    plt.scatter(freq, alpha, s=5, c=magnitude, cmap='turbo')
    plt.xlabel('Spectral Freq (%sHz)' % pre)
    plt.ylabel(r'Cycle Frequency $\alpha$ (%sHz)' % pre)
    cbar = plt.colorbar(aspect=50)
    cbar.set_label('log$_{10}$(SCF)')

    f0 = min(freq)
    f1 = max(freq)
    gap = 0.02*(f1 - f0)
    plt.xlim(f0-gap, f1+gap)

    a0 = min(alpha)
    a1 = max(alpha)
    gap = 0.02*(a1 - a0)
    plt.ylim(a0-gap, a1+gap)

    if fout == None:
        plt.show()
    else:
        plt.savefig(fout, bbox_inches='tight', format='png')

def scf(scfData, key=None, fc_Hz=0, fs_Hz=1, fout=None):

    # Scale frequency (X axis).
    _, pwr10, _ = util.engNot(fs_Hz)
    pre = util.siPrefix(pwr10)
    n = scfData[0].size # Points in each SCF curve.
    f = (fs_Hz*np.fft.fftshift(np.fft.fftfreq(n)))/10**pwr10

    # Find max SCF value (Y axis) of all curves.
    curves = []
    ymax = None
    for i in range(len(scfData)):
        curve = scfData[i]
        curve[curve == 0] = 1e-20 # Avoid log(0).
        curve = util.W_to_dBW(np.abs(curve))
        if ymax == None: # First time through loop.
            ymax = max(curve)
        else:
            ymax = max(ymax, max(curve))
        curves.append(curve)

    #
    #   P l o t   D a t a
    #
    fig, ax = plt.subplots()
    ax.set_box_aspect(1/2) # Graph height is 1/2 x width.
    plt.grid(True)
    plt.xlabel('Spectral Freq (%sHz)' % pre)
    plt.ylabel('Magnitude (dB)')

    span = 20 # dB on vertical axis.
    plt.ylim(ymax-span, ymax + 0.05*span)
    for i in range(len(curves)):
        plt.plot(f, curves[i], linewidth=0.5)
    if key != None:
        plt.legend(key)
    if fout == None:
        plt.show()
    else:
        plt.savefig(fout, bbox_inches='tight', format='png')

def slices(alphas, curvesZ, fc_Hz=0, fs_Hz=1, fout=None, az=125, el=20,
    zlabel='Coherence'):

    #
    #   P r e p a r e   D a t a
    #
    _, pwr10, _ = util.engNot(fs_Hz)
    pre = util.siPrefix(pwr10)
    # Create x axis spectral frequency values.
    n = curvesZ[0].size # Points in each slice.
    f = np.fft.fftshift(np.fft.fftfreq(n))
    # Pre- and post-pend adding room for polygon edges, below.
    f = np.concatenate(([f[0]-1/n], f, [f[-1]+1/n]))
    f *= fs_Hz/10**pwr10 # Convert normalized to absolute frequencies.
    a = (alphas*fs_Hz)/10**pwr10 # Absolute cycle frequencies.

    # Convert to |curvesZ|.
    curves = []
    for c in range(len(curvesZ)):
        curves.append(np.abs(curvesZ[c]))
    curves = np.array(curves)

    #
    #   C r e a t e   P o l y g o n   S l i c e s
    #

    # Find largest SCF values in all SCF curves.
    zMax = curves[0][0] # Initialize to any element.
    zMin = curves[0][0]
    for z in range(len(curves)): # Find max SCF value.
        zMax = max(zMax, np.max(curves[z]))
        zMin = max(zMin, np.max(curves[z]))

    # Create polygonal slices.
    colors = plt.cm.jet(np.linspace(0, 1, a.size)) # Rainbow.
    verts = [] # Each verts[i] is a list of polygon (x,z) vertices.
    for z in range(len(curves)): # Loop through each data slice.
        poly = np.concatenate(([0], curves[z], [0]))
        verts.append(list(zip(f, poly)))            # (x,z) vertices.
    poly = PolyCollection(verts, facecolors=colors) # Colored faces.
    poly.set_linewidth(0.2)                         # Thin edges.
    poly.set_edgecolor('#000000')                   # Black edges.
    poly.set_alpha(0.8)                             # Slightly transparent.

    #
    #   P l o t   S l i c e s
    #
    ax = plt.figure().add_subplot(projection='3d')
    ax.view_init(el, az)
    ax.add_collection3d(poly, zs=a, zdir='y')
    plt.rcParams['axes.grid'] = True
    ax.set_xlabel(r'$f$, spectral freq (%sHz)' % pre)
    ax.set_ylabel(r'$\alpha$, cycle freq (%sHz)' % pre)
    ax.set_zlabel(zlabel)
    ax.set_ylim3d(fc_Hz-fs_Hz, fc_Hz+fs_Hz)
    ax.set_xlim3d(min(f), max(f))
    ax.set_ylim3d(min(a), max(a))
    ax.set_zlim3d(0, zMax)
#   ax.set_zlim3d(zMax-20, zMax)
#   ax.set_zlim3d(0, 1)
    ax.set_box_aspect(None, zoom=1.25) # Don't cut off vertical axis label.
    if fout == None:
        plt.show()
    else:
        # plt.savefig(fout, bbox_inches='tight', format='png')
        plt.savefig(fout, format='png')

def spectrumDual(iq, fc_Hz=0, fs_Hz=1, fout=None):
    n = iq.size
    spectrum = np.abs(np.fft.fft(iq))/n
    spectrum = np.fft.fftshift(spectrum) # DC centered spectrum.
    spectrum = csp.smooth(int(0.01*spectrum.size), spectrum)
    spectrum[spectrum==0] = 1e-200 # Avoid log10(0).
    spectrum = 20*log10(spectrum) # Convert to dB.

    bw_MHz = (fs_Hz/1e6)/2 # Half bandwidth, max freq is fs_Hz.
    freqs_MHz = np.linspace(-bw_MHz, bw_MHz, n)
    freqs_MHz += fc_Hz/1e6 # Move to center frequency.
    len_us = (n/fs_Hz)*1e6 # Signal length in microseconds.
    time_us = np.linspace(0, len_us, n) # X axis for time domain plot.

    plt.rcParams['axes.grid'] = True
    f, ax = plt.subplots(2, 1, figsize=(7,7))
    plt.subplots_adjust(hspace=0.7) # Room for lower subplot title.

    # Time domain plot.
    ax[0].plot(time_us, iq.real, '#red', linewidth=0.75)
    ax[0].plot(time_us, iq.imag, '#blue', linewidth=0.75)
    ax[0].set_title('\n\nTime Domain')
    ax[0].set_xlabel('Time (us)')
    ax[0].set_ylabel('Magnitude (linear)')

    # Frequency domain plot.
    ax[1].set_title('Frequency Spectrum')
    ax[1].plot(freqs_MHz, spectrum, '#449f29', linewidth=0.75)
    ax[1].set_xlabel('Frequency (MHz)')
    ax[1].set_ylabel('Magnitude (dB)')

    if fout == None:
        plt.show()
    else:
        plt.savefig(fout, bbox_inches='tight', format='png')

def spectrum(iq, fc_Hz=0, fs_Hz=1, foutTime=None, foutFreq=None):

    n = iq.size

    #
    #   P l o t   T i m e   D o m a i n   i n   u s
    #
    len_us = n/fs_Hz # Signal length in seconds.
    _, pwr10, _ = util.engNot(len_us)
    pre = util.siPrefix(pwr10)
    time_us = np.linspace(0, len_us, n)/10**pwr10

    fig, ax = plt.subplots()
    ax.set_box_aspect(1/2) # Graph height is 1/2 x width.
    plt.grid(True)
    plt.xlabel('Time (%ss)' % pre)
    plt.ylabel('Magnitude (linear)')
    plt.plot(time_us, iq.real, '#2840b7', linewidth=0.5)
    if foutTime == None:
        plt.show()
    else:
        plt.savefig(foutTime, bbox_inches='tight', format='png')

    #
    #   P l o t   F r e q   S p e c t r u m
    #
    _, pwr10, _ = util.engNot(fs_Hz)
    pre = util.siPrefix(pwr10)

    psd = util.psd(iq) # Calculate PSD.
    f = fs_Hz*np.fft.fftshift(np.fft.fftfreq(psd.size))
    f /= 10**pwr10

    fig, ax = plt.subplots()
    ax.set_box_aspect(1/2) # Graph height is 1/2 x width.
    plt.grid(True)
    plt.xlabel('Frequency (%sHz)' % pre)
    plt.ylabel('Magnitude (dB)')
#   plt.ylim(ymin, ymax)
    plt.plot(f, psd, '#449f29', linewidth=0.5)
    if foutFreq == None:
        plt.show()
    else:
        plt.savefig(foutFreq, bbox_inches='tight', format='png')

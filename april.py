#!/usr/bin/env python

'''
Generate two BPSK signals with slightly different center frequencies and
different symbol rates.  This provides a nice example to show off
CSP's abilities.  Values from example in Eric April's 1991 paper, "The
Advantages of Cyclostationary Processing."

The dual BPSK signal is plotted in 3d as SCF slices, nicely making PSD, and
each BPSK signal apparent at the different cycle frequencies.

Mike Markowski
mike.ab3ap@gmail.com
June 10, 2026
'''

import mksig
import cspPlot
import libcsp as csp
import numpy as np

s = mksig.bpskDual() # Get the IQ!
scnc = [] # Spectral coherence for given alpha.
alphas = np.linspace(-12, 12, 25)/32 # Cheating, we know 1/32 cycles.
for alpha in alphas:
    scf = csp.scfTsm(s, 128, alpha)
    scnc.append(scf)
cspPlot.slices(alphas, scnc, zlabel='SCF')

# This code implements mathematics presented by Chad Spooner in
# his blog at https://cyclostationary.blog/ 
#
# SSCA code is implemented from Eric April's paper, "On the Implementation
# of the Strip Spectral Correlation Algorithm for Cyclic Spectrum Estimation"
# by Eric April.  1995.
#
# Subroutines are in alphabetical order.  A good deal of work has gone into
# removing all 'for' loops for improved vectorization of code, spending more
# time in the faster numerical C code underlying numpy and scipy.  Similar
# benchmarking has been used to determine when to use one approach or
# another.
#
# Despite time taken to make the code efficient, more time has been spent to
# make this closely follow cyclostationary.blog and retain a high level of
# readability.  The goal is educational more than efficiency.
#
# Variable names use format thing_units, where underscore separates name from
# its units.  E.g., fc_Hz is carrier freq in Hz.
#
# Mike Markowski
# mike.ab3ap@gmail.com
# Mar 2021 original
# Jun 2026 extensive rewrite

from numpy import log2, pi, sqrt # Commonly used functions.
from numpy.lib.stride_tricks import sliding_window_view
from scipy import fft, ndimage, signal
from scipy.interpolate import interp1d
import numpy as np
import util

def alphaRefine(sig, alphas, conj=False, radius=2, pulseWidth=0):
    '''Refine blind detected cyclic frequency estimates onto the grid
    that cyper()/shifts() needs to produce a matched (a1==a2) shift.
    This subroutine is slooow.

    Blind detectors (SSCA, FAM) locate alpha to within a bin or two of true
    cyclic frequency, but shifts() only achieves a1==a2 for roughly half
    of each 1/sig.size wide cell around a candidate (see shifts()).  For
    signals with sharp, low jitter cyclic features, like clean test signals,
    landing in the wrong half of that cell is the difference between finding
    the true peak and finding noise floor.  It's worse for FAM alphas since
    fam() builds its alpha grid on a power of 2 zero padded copy of the
    signal, so its grid doesn't line up with 1/sig.size.

    This does a tiny local search on the native 1/sig.size grid: for each
    alpha, try nearby grid points and keep whichever gives the largest
    cyclic periodogram magnitude.

    cyper() is a single snapshot, unsmoothed periodogram estimate.  It has
    high variance across nearby unrelated alpha values, and like an ordinary
    periodogram is an inconsistent PSD estimator.  Scoring candidates on
    np.max() of that raw magnitude over all n frequency bins compares noisy
    extreme value statistics between alphas, which can let a neighboring,
    wrong grid point outscore the true cyclic frequency by chance.  Smoothing
    trades a little frequency resolution for a lower variance comparison,
    letting the true peak win.

    Inputs:
      sig (complex[]): IQ signal.
      alphas (float[]): cyclic frequencies to refine.
      conj (boolean): default False, conjugate/non-conjugate.
      radius (int): default 2, how many 1/sig.size grid points to try
        on either side of each alpha.
      pulseWidth (int): smoothing pulse width, applied to each periodogram
        before scoring.  Default 0 sets it to sig.size//100, matching
        sc()'s default.
    Output:
      (float[]): refined alphas, same length/order as input.
    '''

    n = sig.size
    if pulseWidth == 0:
        pulseWidth = n//100 # Match sc()'s default smoothing.
    X = cyperInit(sig) # Compute FFT once and reuse.
    out = np.zeros_like(alphas)
    for i, a in enumerate(alphas):
        k0 = round(a*n) # Nearest grid point, units of 1/n.
        bestA, bestMag = a, -1
        for k in range(k0-radius, k0+radius+1):
            aTry = k/n
            mag = np.max(np.abs(smooth(cyper(X, aTry, conj), pulseWidth)))
            if mag > bestMag:
                bestA, bestMag = aTry, mag
                if conj or not np.isclose(bestA, 0):
                    out[i] = bestA
    return out

def binArrays(x, y, epsilon):
    '''
    Bin ascending x[] such that all elements in a single bin are
    within epsilon of each other, i.e., x[last] - x[first] <= epsilon.
    Of those, retain the x[i] whose y[i] is largest within each bin.
    '''

    if len(x) == 0:
        return [], []
    ind = []
    start = 0
    iBest = 0
    for i in range(1, len(x)):
        if x[i] - x[start] <= epsilon: # Fits in bin.
            if y[i] > y[iBest]: # Check best y.
                iBest = i
        else:
            ind.append(iBest)
            start = i
            iBest = i
    ind.append(iBest)  # Final bin.
    return ind

def binAlpha(fs_Hz, nSa, alphas, scfs, n, conj=True, guard=5):
    '''For unique alphas, find maximum SCF for each.  Alphas within 1/nSa of
    each other are binned in binArray() and then the top 2n candidates are
    retained.  While the parameter is named scfs it can be spectral
    coherences are any other ranking choice.  2n are retained because there
    might still be unbinned values that will be binned later for plotting,
    losing some values.  Saving 2n makes it likely there will be at least n,
    as wanted.

    It is important to call refineAlphas() after this routine is called.
    See that routine's documentation for reason why.

    Every alphas[i] corresponds to scfs[i].

    Inputs:
      alphas (float[]): Hz, non-unique cyclic frequencies.
      scfs (float[]): scfs[i] is SCF at alphas[i].
      n (int): return at most n (actually, 2*n, due to hack) alphas and SCFs
        from max SCF onward.
      conj (boolean): True/False for conjugate/non-conjugate.
      guard (int): number of bins that non-conj PSD energy might leak into.
    Outputs:
      alphasUnique, scfs (float[], float[]): unique alphas and their max SCF.
    '''

    epsilon = 1/nSa # Freq resolution.
    n *= 2 # XXX Hack!  Get double needed to later trim down.

    ind = np.argsort(alphas) # Sort by x ascending.
    alphas = alphas[ind]
    scfs = scfs[ind]
    ind = binArrays(alphas, scfs, epsilon)
    alphas = alphas[ind]
    scfs = scfs[ind]

    if conj:
        ind = np.argsort(scfs)[::-1] # Sort by descending SCF.
        ind = ind[:n] # Top n SCFs.
        topA = alphas[ind]
        topS = scfs[ind]
        ind = np.argsort(topA) # Sort by ascending alpha.
        topA = topA[ind]
        topS = topS[ind]
    else:
        # Drop PSD (alpha=0) and its skirt: blind-detected alpha
        # imprecision spreads PSD energy across more than one bin
        # around zero, so a single bin exclusion isn't enough.
        # Exploit symmetry, keep alpha > 0 side only, then rank by
        # magnitude same as conjugate branch.
        mask = alphas > guard*epsilon
        alphas = alphas[mask]
        scfs = scfs[mask]
        ind = np.argsort(scfs)[::-1] # Sort by descending SCF.
        ind = ind[:n] # Top n SCFs.
        topA = alphas[ind]
        topS = scfs[ind]
        ind = np.argsort(topA) # Sort by ascending alpha.
        topA = topA[ind]
        topS = topS[ind]
    return topA, topS # Unique alphas and max SCFs.

def cyperInit(sig, padFactor=1):
    '''Set up prior to cyclic periodogram.  This separates the expensive FFT
    from the rest of the algorithm, helpful when the same signal is studied
    with many alphas.

    Inputs:
      sig (complex[]): time domain signal of interest.
      padFactor (int): Chad Spooner illustrates benefits of zero padding
        at https://cyclostationary.blog/2021/05/05/zero-padding-in-spectral-correlation-estimators/#more-10253
        where he says that a factor of 2 or 4 is best.  More bins allow
        finer frequency discernment.
    Output:
      (complex[]): FFT of sig, zero padded if user requested.
    '''

    x = np.pad(sig, (0, sig.size*(padFactor-1))) if padFactor > 1 else sig
    return fft.fftshift(fft.fft(x))

def cyper(X, alpha, conj=False):
    '''Cyclic Periodogram as described at:

    https://cyclostationary.blog/2015/11/20/csp-estimators-the-frequency-smoothing-method/

    The subroutine implements equations 8 (non-conjugate) and 11 (conjugate).

    Inputs:
      X (complex[]): frequency domain signal of interest.
      alpha (float): cycle frequency for analysis.
      conj (boolean): True conjugate/non-conjugate cyclic periodogram.
    Output:
      (complex[]): SCF of sig at cyclic freq alpha.
    '''

    n = X.size
    a1, a2 = shifts(alpha, n, conj)
    Xup = zshift(X, -a1)
    Xdn = zshift(X[::-1], a2) if conj else zshift(X.conjugate(), a2)
    return Xup*Xdn/n # Cyclic periodogram.

def fam(sig, L, Np=0, conj=False, showMults=False):
    '''FFT Accumulation Method, computes spectral correlation estimates
    over its entire principal domain.  It does the same thing as ssca().
    Alpha estimates should be refined with alphaRefine() before use.  See
    its documentation for info.

    [1] Implementation based on 'Computationally Efficient Algorithms
    for Cyclic Spectral Analysis,' Roberts, Brown and Loomis, IEEE SP
    Magazine, Apr 1991.

    Also, drawn from:

    [2] https://cyclostationary.blog/2018/06/01/csp-estimators-the-fft-accumulation-method/

    [3] "Implementation of Cyclic Spectral Analysis Methods," LCDR Nancy J.
    Carter, 1992, Naval Postgraduate School.

    Equation numbers come from the blog page, [2], above.

    Inputs:
      sig (complex[]: signal of interest.
      L (int, dyadic): hop length, in units of samples.
      Np (int, dyadic): length of FFTs.  N'=4L is recommended.
      conj (boolean): default False, calculate conjugate/non-conjugate
        cyclic periodogram.
      showMults (boolean): default False, when True prints information on
        numbers of multiplications.  Only useful when comparing algorithms.
    Outputs:
      f_j,   (float[][1]): spectral frequencies,
      alpha, (float[Np**2,1]): cyclic freqs, and
      Sx,    (float[Np**2,P]): spectral correlation estimate.

    Ex.
     N = 65536
     N' = 32
     L = 8
    '''

    N = 2**int(log2(sig.size))
    if N != sig.size: # Zero pad to power of 2 size.
        N <<= 1
        x = np.zeros(N, dtype=complex)
        x[:sig.size] = sig
        sig = x
    if Np == 0:
        Np = 4*L                      # Recommended trade-off in [1].
    if not famReady(sig[:N], L, Np):  # Assortment of sanity checks.
        return []

    P = N//L                          # Eq 4, will do Np P-point FFTs.
    pad = Np + L*(P - 1) - N          # Pad x() with to fill matrix.
    x = np.append(sig, np.zeros(pad)) # x is zero padded sig.
    y = x.conjugate() if conj else x
    N = x.size                        # Updated signal length.

    # Step 1: Arrange N'-point Data Sub-blocks.
    # [x(0)  x(L+0)  x(2L+0)  ...  x((P-1)L + 0)]
    # [x(1)  x(L+1)  x(2L+1)  ...  x((P-1)L + 1)]
    #                         ...
    # [x(N'-1) x(L+N'-1) x(2L+N'-1) ...  x((P-1)L + N'-1)]
    colNp = np.arange(Np).reshape((Np, 1))
    rowP = np.arange(P)
    X = x[colNp + L*rowP] # Np x P, each col is x' staggered by L samples.
    Y = y[colNp + L*rowP] if conj else X

    # Step 2: Apply Data Tapering Window to Columns of X.
    A = signal.get_window('hamming', Np) # a(r) in Eq 3.
    A = A.reshape((Np, 1)) # Np x 1.
    XA = X*A               # Np x P, tapered.
    YA = Y*A if conj else XA

    # Step 3: Apply Fourier Transform to Windowed Subblocks.
    XAT = fft.fftshift(fft.fft(XA, axis=0), axes=0) # Column FFTs, Eq 3.
    YAT = fft.fftshift(fft.fft(YA, axis=0), axes=0) if conj else XAT
    # Mix to baseband.
    f = fft.fftshift(fft.fftfreq(Np)) # Spectral components of XAT.
    f = f.reshape((Np, 1)) # Np x 1.
    q = np.arange(P)*L     # 1 x P, scale frequency by stagger.
    E = np.exp(-2j*pi*f*q) # Np x P, phase adjustment.
    Xg = XAT*E             # Np x P, demodulates mixed to baseband.
    Yg = YAT*E if conj else Xg

    # Step 4: Multiply Channelized Subblocks Together and FFT.
    # Sx: spectral correlation estimate, Np x P matrix.
    Ygc = Yg.conj()                 # Significantly speeds up next line.
    Sx = Xg[:,None,:]*Ygc[None,:,:] # (Np,1,P)*(1,Np,P) => (Np,Np,P)
    Sx = Sx.reshape((Np*Np, P)) # Collapse P pages of (Np,Np) to (Np**2,P).
    Sx = fft.fftshift(fft.fft(Sx), axes=1) # Row FFTs.
    Sx /= P # Rectangular smoothing window.
    Sx *= 3/2*Np/np.sum(A**2) # Compensate for Hamming window's power loss.
    e = P//4
    Sx = Sx[:, e:3*e] # Save center half of FFT, locations 1/4 to 3/4.

    # Step 5: Associate Fourier Transform Outputs with Freqs.
    # Spectral (f_j) and cycle (alpha_i) frequencies.
    f_j = (f[:, None] + f[None, :])/2     # Eq 7.
    f_j = f_j.reshape((Np*Np, 1))         # Np*Np x 1.
    alpha_i = f[:, None] - f[None, :]     # Eq 6.
    alpha_i = alpha_i.reshape((Np*Np, 1)) # Np*Np x 1.
    q = np.arange(-P//4, P//4)            # Cycles about alpha_i.
    if P%2 == 0: # Even.
        q += 1 # Compensate for asymmetry.
    alpha = alpha_i + q/N                 # Np*Np x P, Eqs 5 & 8.

    if showMults: # Algorithmic info, generally not wanted.
        # From [3], p. 12.
        m = (6 + 4*Np)*P*Np + (2*P*Np)*(log2(Np) + Np*log2(P))
        print('fam info: FAM multiplications: %.1f (x1e6)' % (m/1e6))

    return f_j, alpha, Sx # Np*Np x 1, Np*Np x P, Np*Np x P.

def famReady(x, L, Np):
    '''This routine performs a few sanity checks to make we're ready to
    perform an actual FAM.  Warnings and errors are printed.  In the case of
    warnings the routine returns True, meaning things can move forward.
    False means everything must stop.  Raising an exception is possibly
    better, but this is educational code so isn't done.

    Inputs:
      x (complex[]): signal of interest.
      L (int, dyadic): hop length, in units of samples.
      Np (int, dyadic): default 0 (unset), length of FFTs.  N'=4L recommended.
    Output:
      (boolean): True if safe to proceed with FAM, False if not.  Warnings
        yield a True return, errors False.
    '''

    # Some sanity checks.
    if L > Np:
        print('famReady error: Need L < N\', but %d > %d.' % (L, Np))
        return False
    elif L == Np:
        s = 'famReady warning: L == N\' == %d yields substantial ' % L
        s += 'cycle leakage.'
        print(s)
        return True
    # Ensure N and L are dyadic.
    N = x.size
    N2 = log2(N)
    L2 = log2(L)
    if not np.isclose([int(L2)], [L2]): # Not exact power of 2.
        print('famReady error: non-dyadic hop length L=%d.' % L)
        return False
    if not np.isclose([int(N2)], [N2]): # Not exact power of 2.
        print('famReady error: non-dyadic signal length N=%d.' % N)
        return False
    return True

def famResolution(N, Np, q):
    '''Used to study algorithm, not called normally.
    '''
    # Resolution of result.  'd' is short for delta, used in paper.
    fs = 1                     # Normalized fs, used to keep equations general.
    dt = N/fs                  # Delta T, length of signal.
    da = fs/Np                 # Resolution determined by tapering window.
    dalpha = fs/N              # Cycle freq resoln deps on points processed.
    df = da - np.abs(q)*dalpha # Freq resolution is function of q.
    dtdf = dt*da               # dt*df when q == 0.
    return df, dalpha, dtdf

def famSc(x, f, alpha, famScf, conj=False):
    '''Convert FAM spectral correlation function (SCF) output to spectral
    coherences (SC).  Variable names and algorithm implemented from

    https://cyclostationary.blog/2018/06/01/csp-estimators-the-fft-accumulation-method

    Inputs:
      x (complex[]): 1d signal whose PSD is used in SC calculations.
      famScf (complex[]): Np x P matrix of SCF value from FAM.
      conj (boolean): default False, calculate conjugate/non-conjugate
        cyclic periodogram.
    Output:
      (complex[][]:  Np x P matrix of spectral coherence values
        corresponding to each element in famScf[][].
    '''

    scf = famScf.copy() # Don't modify user's data.
    Np, P = scf.shape

    # Calculate PSD and function to interpolate it.
    X = cyperInit(x)
    psd = cyper(X, alpha=0)
    psd = smooth(psd)
    fP = fft.fftshift(fft.fftfreq(psd.size)) # PSD spectral components.
    fn = interp1d(fP, psd, fill_value=(psd[0],psd[-1]),
        bounds_error=False, kind='nearest') 

    # Prepare roll up/down values and interpolate rolled PSDs.
    Xdn = f + alpha/2                      # Np x P.
    Xup = alpha/2-f if conj else f-alpha/2 # Np x P.
    dnI = fn(Xdn)         # 1 x P row.
    upI = fn(Xup)         # Np x 1 col.
    denom = sqrt(dnI*upI) # Np x P matrix,

    relFloor = 1e-6*np.median(np.abs(denom))
    z = np.where(np.abs(denom) < relFloor) # Div by 0 (or nearly) locations.
    scf[z] = 0 # Return 0 coherence for divisons by 0.
    denom[z] = 1
    sc = scf/denom # Spectral coherences, Np x P matrix.
    sc[np.abs(sc) > 1.01] = 0 # Remove numerical artifacts.
    return sc # Spectral coherences, Np x P matrix.

def scfFilter(f, alpha, sc, scf, threshold=0, top=0, alpha0=None, sortby='sc'):

    '''Return matrix whose rows are [alpha, f, sc, scf].

    Performs Step 5 from April's paper:
    "On the Implementation of the Strip Spectral Correlation Algorithm for
    Cyclic Spectrum Estimation" by Eric April.  1995.

    Each strip FFT contributes to PSD, (a=0)
    Each FFT has N points, each point a cycle frequency

    Inputs:
      f (float[]): spectral freq
      alpha (float[]): cycle freq
      sc (complex[]): spectral coherence matrix.
      scf (complex[]): spectral correlations corresponding to sc[].

    Outputs:
      ([[f, alpha, scf, sc],]) 
          where
          f, spectral frequency,
          alpha, cyclic frequency,
          scf, spectral correlation,
          sc, coherence.
    '''

    rows, cols = scf.shape    # rows x cols.
    scAbs = np.abs(sc)        # rows x cols, complex to real magnitude.
    scfAbs = np.abs(scf)      # rows x cols.

    if f.shape != scf.shape:  # FAM.
        fp = np.tile(f, cols) # Turn rows x 1 into rows x cols.
        fam = True
    else: # SSCA.
        fp = f
        fam = False

    # Create quads of [spectral freq, cyclic freq, SCF, spectral coh].
    afsc = np.zeros((rows*cols, 4))
    afsc[:, 0] = fp.flatten()
    afsc[:, 1] = alpha.flatten()
    afsc[:, 2] = scfAbs.flatten()
    afsc[:, 3] = scAbs.flatten()
    sortCol = 3 if sortby == 'sc' else 2
    threshCol = 3 if sortby == 'sc' else 2

    # See https://cyclostationary.blog/2018/06/01/csp-estimators-the-fft-accumulation-method/
    if fam: # Don't waste time unnecessarily doing this on ssca results.
        # For FAM, save normalized freqs where f +/- alpha2 in [-0.5, 0.5].
        afsc = afsc[afsc[:,0] + afsc[:,1]/2 >= -0.5]
        afsc = afsc[afsc[:,0] + afsc[:,1]/2 <=  0.5]
        afsc = afsc[afsc[:,0] - afsc[:,1]/2 >= -0.5]
        afsc = afsc[afsc[:,0] - afsc[:,1]/2 <=  0.5]

    if threshold > 0:
        afsc = afsc[afsc[:,threshCol] > threshold] # Ignore low SCs.

    if top == 0: # Return 'top' number of rows.
        top = sc.size

    # afsc: N*rows x 4 rows of [ [f, alpha, scf, sc], ...] .
    if alpha0 is None: # Return 'top' SCF or SC values.
        idx = np.argsort(afsc[:, sortCol])[::-1]  # Descending.
        afsc = afsc[idx[:top]]
    else: # Return rows close to specified alpha.  Usually, for debug plots.
        col = 1 # Col 1 is alpha.
        i = np.where(np.abs(afsc[:,col] - alpha0) < 1/(2*rows))
        afsc = afsc[i]
        afsc = np.ndarray.tolist(afsc) # Can only sort lists.
        afsc.sort(key=lambda row: row[0:]) # Sort by spectral freq.
        afsc = np.array(afsc)
    return afsc

def periodogram(sig):
    '''Estimate a signal's PSD with Daniell method, a frequency-smoothed
    periodogram.  See:

    https://cyclostationary.blog/2015/11/20/csp-estimators-the-frequency-smoothing-method/

    Equation numbers in comments below refer to the above web page.  This
    subroutine implements Eq 2.

    Inputs:
      sig (complex[]): symbol stream whose PSD estimate is wanted.

    Outputs:
      (complex[]): periodogram of sig.
    '''

    I = fft.fft(sig)
    I = fft.fftshift(I) # Move DC to center.
    I = I*I.conjugate()/I.size # Eq 2, |X(f)|^2 / N.
    return smooth(I, 0.005*I.size) # Smoothing window 0.5% of signal.

def sc(sig, alpha, conj=False, pulseWidth=0, padFactor=1):
    '''Calculate spectral coherence of a signal at specified cycle frequency.

    https://cyclostationary.blog/2016/01/08/the-spectral-coherence-function/

    Inputs:
      sig (complex[]): signal to be analyzed.
      alpha (float): cycle frequency.
      conj (boolean): default False, calculate conjugate/non-conjugate
        cyclic periodogram.
      pulseWidth (int): width, in array elements, of smoothing pulse.
        When left at default of 0, is then set to len(sig)//100.
    Output:
      (complex[]): spectral coherence of signal at cycle freq alpha.
    '''

    if pulseWidth == 0:
        pulseWidth = sig.size//100 # Recommended in Chad's blog.

    X = cyperInit(sig, padFactor=padFactor)
    scf = cyper(X, alpha, conj)
    scf = smooth(scf, pulseWidth)
    psd = cyper(X, alpha=0) # SCF for alpha=0 is PSD.
    psd = smooth(psd, pulseWidth)

    a1, a2 = shifts(alpha, psd.size, conj) # Up/down shifts for PSD.
    Xup = zshift(psd, -a1) # Shift left a1 elements and zero fill.
    Xdn = zshift(psd[::-1], a2) if conj else zshift(psd, a2) # Shift right a2.
    denom = sqrt(Xup*Xdn)

    z = np.where(np.abs(denom) < 1e-6) # Div by 0 locations.
    scf[z] = 0 # Return 0 coherence for divisons by 0.
    denom[z] = 1

    return scf/denom # Spectral coherence.

def scfTsm(sig, N, alpha, conj=False, taper=False):
    '''Spectral Correlation Function, Time Smoothing Method.  Implementation
    of:

    https://cyclostationary.blog/2015/12/18/csp-estimators-the-time-smoothing-method/

    Variable names are taken from the blog.  Compute the cyclic periodogram
    for blocks in time domain and average results.

    Inputs:
      sig (complex[]): signal to be analyzed.
      N (int): points per TSM block.
      alpha (float): cycle frequency of interest.
      conj (boolean): want conjugate/non-conjugate cyclic periodogram.
      taper (boolean): default False, whether to use/not use a taper window,
        recommend by Chad Spooner.
    Output:
      (complex[]): SCF of sig at cyclic freq alpha.
    '''

    S = util.zeroPad(sig, N)    # Zero pad till sig is multiple of N.
    M = S.size//N               # M blocks of N points.
    S = S.reshape((M, N))       # Prepare for M rows of N-pt FFTs.
    if taper:                   # Optional taper.
        w = signal.get_window('hamming', N)
        w /= util.rms(w)        # RMS normalized window.
        S *= w                  # Tapered signal.

    I = fft.fftshift(fft.fft(S, axis=1), axes=1) # Row FFTs.
    a1, a2 = shifts(alpha, N, conj) # Cyclic shifts.
    Xup = zshift(I, -a1)        # Upward shift.
    Xdn = np.flip(I, axis=1) if conj else I.conjugate()
    Xdn = zshift(Xdn, a2)       # Downward shift.
    I = Xup*Xdn/N               # Eq 3, each row is a cyclic periodogram.

    u = (np.arange(M)*N)[:,None]
    I *= np.exp(-2j*pi*alpha*u) # Phase compensation.
    S = np.sum(I, axis=0)       # Sum columns, frequency components.
    return S/M                  # Eq 5, SCF estimate.

def scfTsmLoop(sig, N, alpha, conj=False, taper=False):
    '''XXX Works, but much slower than matrix version.  Timings below are
    from an older version of this code.  Current loop vs matrix timings are
    reasonably close, thanks to general clean up of efficiency issues.
    It's still better to use the matrix version, though, which will likely
    take advantage of future python improvements.

    Typical comparison:

        64 point TSM ffts
          Loop:   82.3 ms
          Matrix:  1.4 ms,    Speed up: 57.3x

        64 point TSM ffts
          Loop:   85.2 ms
          Matrix:  1.7 ms,    Speed up: 50.2x

    Spectral Correlation Function, Time Smoothing Method.  Implementation
    of:

    https://cyclostationary.blog/2015/12/18/csp-estimators-the-time-smoothing-method/

    Compute the cyclic periodogram for blocks in time domain and average
    results.
    '''

    print('Loop TSM')
    M = sig.size//N    # M blocks of N points.
    S = np.zeros(N, dtype=complex) # Eventually is the desired SCF.
    for i in range(M): # Loop through M segments in time.
        u = i*N        # Left edge of i'th subblock.
        X = cyperInit(sig[u:u+N])
        I = cyper(X, alpha, conj) # u'th cyclic periodogram.
        I *= np.exp(-2j*pi*alpha*u)
        S += I
    return S/M

def shifts(alpha, n, conj=False):
    '''
    Choose shifts 'a1' and 'a2' such that is minimized.  Calculate upward
    and downward shifts for cyclic periodogram by minimizing
    |alpha - (|a1/n| + |a2/n|).  Blog reference:

    C. M. Spooner and R. B. Nicholls, “Spectrum Sensing Based on Spectral
    Correlation,” Chapter 18 in Cognitive Radio Technology, Second Edition,
    Ed. Bruce Fette, 2009.

    Inputs:
      alpha (float): cyclic frequency.
      n (int): signal length.
      conj (boolean): default False, calculate shifts for conjugate
        or non-conjugate.

    Outputs:
      (int, int): a1, a2 (upward, downward) shifts.
    '''

    if alpha == 0: # PSD.
        a1 = a2 = 0
    else:
        f = int(np.floor(alpha*n/2)) # Amount to shift upward,
        c = int(np.ceil (alpha*n/2)) # and amount to shift downward.
        aa = abs(alpha*n)
        af = abs(f)
        ac = abs(c)
        fc = abs(aa - (af + ac))
        ff = abs(aa - (af + af))
        cc = abs(aa - (ac + ac))
        m = min(fc, ff, cc)
        # Adjust shifts, accounting for discrete index offsets.
        if m == fc:
            a1,a2 = f,c
        elif m == ff:
            a1,a2 = f,f
        else: # m == cc
            a1,a2 = c,c
    if conj and n%2 == 0:
        a2 += 1 # Adjust for flipping even length array.
    return a1, a2 # Up, down shifts.

def smooth(sig, wid=None):
    '''Smooth a signal with a unit area pulse.  Eq 3 in:

    https://cyclostationary.blog/2015/11/20/csp-estimators-the-frequency-smoothing-method/

    Inputs:
      sig (complex[]): signal to smooth.
      wid (int): units of array elements, pulse width to smooth.
    Output:
      (complex[]): smoothed signal.
    '''

    if wid is None:
        wid = 0.01*sig.size
    wid = int(wid)
    if wid <= 1:
        return sig.copy()

#   pulse = np.ones(wid)/wid # Unit area rectangle.
#   res = np.convolve(pulse, sig, mode='same') # Smoothed signal.
#   return res

    # Speedier version of commented out code above.
    re = ndimage.uniform_filter1d(sig.real, size=wid, mode='constant')
    im = ndimage.uniform_filter1d(sig.imag, size=wid, mode='constant')
    return re + 1j*im

def ssca(x, Np, N=None, conj=False, showMults=False):
    '''Calculate Strip Spectral Correlation Analyzer of a signal.  From:

    Implementation directly from Section 3.2, steps for matrix approach.
    Variables in code mirror those in paper:

    [1] "On the Implementation of the Strip Spectral Correlation Algorithm
    for Cyclic Spectrum Estimation" by Eric April.  1995.

    [2] "Implementation of Cyclic Spectral Analysis Methods," LCDR Nancy J.
    Carter, 1992, Naval Postgraduate School.

    Inputs:
      x (complex[]): signal to perform SSCA on. Must be N+Np samples long.
      N (int): number of points to analyze.
      Np (int): number of channels in the channelizer.
      conj (boolean): default False, take conjugate/non-conjugate SSCA.
      showMults (boolean): default False, show algorithmic info.

    Outputs:
      (complex[][]): N x Np SCF of input of signal.
    '''

    # Step 1.  Create sliding windowed vector of input signal.
    if N is None:
        N = x.size - Np
    if x.size < N + Np: # Quick error check.
        print('ssca warning: signal length %d < N+Np=%d.'
            % (x.size, (N+Np)), end='')
        N = x.size - Np
        print('  Setting N to %d.' % N)

    # Construct X, Eqn 26.
    # The transpose in
    #    X = np.array([x[i:i+N] for i in range(Np)]).transpose()
    # leaves matrix non-contigous and subsequent FFTs are 10x slower!
    # Slow XXX X = np.array([x[i:i+Np] for i in range(N)]) # Staggered rows.
    X = sliding_window_view(x, Np)[:N] # Staggered rows.

    # Step 2. N (row) FFTs of Np points.
    A = signal.get_window(('chebwin', 96), Np) # Tapering window, 1 x Np vec.
    A /= np.max(A)
    XAT= fft.fftshift(fft.fft(X*A), axes=1) # Row FFTs, Eqn 27.

    # Step 3.  Phase shifts.
    k = (np.arange(Np) - Np/2)/Np   # 1 x Np.
    n = np.arange(N).reshape((N,1)) # N x 1.
    E = np.exp(-2j*pi*k*n)          # N x Np matrix, Eqn 29.

    # Step 4.  Np strip (column) FFTs of N points.
    Xg = XAT*E                    # Np x P, demodulates mixed to baseband.
    i = Np//2 # Start index of signal centered in N+Np samples.
    sig = x[i:i+N].reshape((N,1)) # Eqn 30.
    if not conj:
        sig = sig.conjugate()     # Eqn 30.
    Xg *= sig/N # Eqn 31.  '/N' is rect smoothing window of ampl 1/N.
    Sx = fft.fftshift(fft.fft(Xg, axis=0), axes=0) # Col FFTs, Eqn 32.

    # XXX SSCA Resolution of result.  Unsure how useful this is...
#   dt = N # Delta T, number of points in time domain signal.
#   df = 1/Np # Freq resolution depends on tapering window bandwidth.
#   da = 1/dt # Cycle freq resoln deps on points processed.
#   dtdf = dt*df

    if showMults:
        # From [2], p.34.
        L = 1
        m = 2*Np*((6*N/L + 4*N) + (2*N/L + 2*N)*log2(N))
        print('ssca info: SSCA multiplications: %.1f (x1e6)' % (m/1e6))

    q = (np.arange(N) - N/2).reshape((N,1)) # N x 1 column vector.
    k = np.arange(Np) - Np/2 # 1 x Np row vector.
    f = (k/Np - q/N)/2       # N x Np matrix.
    alpha = k/Np + q/N       # N x Np matrix.

#   return Sx, df, da, dtdf # SSCA SCF complex matrix, N x Np.
    return f, alpha, Sx # SSCA SCF complex matrix, N x Np.

def sscaSc(x, sscaScf, conj=False, M=64, taper=False):
    '''Convert SSCA spectral correlation function (SCF) output to spectral
    coherences (SC).

    Inputs:
      x (complex[]): 1d signal whose PSD is used in SC calculations.
      sscaScf (complex[]): N x Np matrix if SCF value from SSCA.
      M (int): SCF TSM block size, default 64.
      conj (boolean): default is False, take conjugate/non-conjugate SSCA.
    Output:
      (complex[][]:  N x Np matrix of spectral coherence values
        corresponding to each element in sscaScf[][].

    '''

    scf = sscaScf.copy() # Don't change user's data.
    N, Np = scf.shape

    # Prepare roll up/down values.
    #
    # Calculate shift values,  f +/- alpha/2.
    # q = {0,1,...,N-1} - N/2      k = {0,1,...,Np-1} - Np/2.
    # f = (k/Np - q/N)/2           alpha = k/Np + q/N
    # dn = f + alpha/2 = k/Np      up = f - alpha/2 = -q/N
    q = np.arange(N) - N/2     # 1 x N, to be reshaped into column later.
    k = np.arange(Np) - Np/2   # 1 x Np row vector.
    dn = k/Np                  # 1 x Np, f + alpha/2.
    up = q/N if conj else -q/N # 1 x N, +/-(alpha/2 - f).

    # Generate PSD for coherence denominator.
    i = Np//2      # Start index of signal.
    sig = x[i:i+N] # 1 x N, N samples centered in N+Np samples.
    psd = scfTsm(sig, M, alpha=0, taper=taper)

    fCoarse = fft.fftshift(fft.fftfreq(M)) # Freqs in PSD.
    fn = interp1d(fCoarse, psd, fill_value=(psd[0],psd[-1]),
        bounds_error=False, kind='nearest') 
    dnI = fn(dn)                 # 1 x Np row.
    upI = fn(up).reshape((N, 1)) # N x 1 col.
    denom = sqrt(dnI*upI)        # N x Np matrix,

    z = np.where(np.abs(denom) < 1e-10) # Div by 0 (or nearly) locations.
    scf[z] = 0 # Return 0 coherence for divisons by 0.
    denom[z] = 1
    return scf/denom # Spectral coherences, N x Np matrix.

def zshift(x, n):
    '''Right shift array by n elements and zero fill.  For example,

    sig = np.array([1,2,3,4,5])
    zshift(sig, 2) => array([0, 0, 1, 2, 3])
    zshift(sig, -3) => array([4, 5, 0, 0, 0])

    Importantly, it also works for two dimensions as needed by CSP code.

    Inputs:
      x (np.array[]): array to roll right and zero fill.
      n (int): number of positions to roll array to right.

    Output:
      (np.array[]): rolled and zero filled array.
    '''

    y = np.zeros_like(x)
    if n >= 0:
        y[..., n:] = x[..., :x.shape[-1]-n]
    else:
        y[..., :n] = x[..., -n:]
    return y


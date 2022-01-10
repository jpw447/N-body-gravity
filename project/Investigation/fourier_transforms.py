import numpy as np

def fourier_func(signal_amplitude:np.ndarray, t_max:float):
    '''Calculates the power and equivalent frequency for the first half of the k-modes for the supplied signal amplitude.  
    Args:
        signal_amplitude (np.ndarray): The supplied signal amplitudes, uniformly sampled from t = 0 to t = t_max
        t_max (float): The maximum time
    Returns:
        (np.ndarray, np.ndarray): A tuple of numpy arrays, the first being the power values and the second being the freqencies corresponding to the power.
    '''
    global FT, frequencies              # Necessary for the inverse Fourier Transform
    FT = np.fft.fft(signal_amplitude)
    N = len(signal_amplitude)
    half = int(N/2)
    FT = FT[:half]                      # Second half of array is irrelevant
    step = 1/t_max
    nyq = half*step                     # Nyquist frequency 
    
    frequencies = np.arange(0, nyq, step)
    FT_conj = FT.conj()
    power = ((FT*FT_conj)/N).real       # Power spectrum from FT×FT*
    
    return power, frequencies
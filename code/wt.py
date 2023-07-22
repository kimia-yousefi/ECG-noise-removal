import numpy as np
import pywt
import matplotlib.pyplot as plt


# ===========================
# 1. Function to add Gaussian noise to a signal
# ===========================
def add_noise(signal, noise_level=0.1):
    """
    Add Gaussian noise to a 1D signal.
    
    Parameters:
        signal: original signal (1D numpy array)
        noise_level: standard deviation of Gaussian noise
        
    Returns:
        noisy signal
    """
    noise = noise_level * np.random.randn(len(signal))
    return signal + noise


# ===========================
# 2. Generate a synthetic ECG signal
# ===========================
def generate_ecg(length=1000, fs=500):
    """
    Generate a simple synthetic ECG-like signal for demonstration.
    
    Parameters:
        length: number of samples
        fs: sampling frequency in Hz
        
    Returns:
        synthetic ECG signal
    """
    t = np.linspace(0, length / fs, length)
    ecg = 1.2 * np.sin(2 * np.pi * 1.33 * t) + 0.25 * np.sin(2 * np.pi * 0.5 * t)
    return ecg


# ===========================
# 3. Wavelet-based denoising
# ===========================
def wavelet_denoising(noisy_signal, wavelet='db4', level=4, thresholding='soft'):
    """
    Denoise a 1D signal using wavelet decomposition.
    
    Parameters:
        noisy_signal: input noisy signal
        wavelet: type of wavelet (default 'db4')
        level: decomposition level
        thresholding: 'soft' or 'hard' thresholding
        
    Returns:
        denoised signal
    """
    coeffs = pywt.wavedec(noisy_signal, wavelet, level=level)
    # Estimate noise sigma using the detail coefficients at level 1
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(noisy_signal)))
    
    # Apply thresholding to all coefficients
    denoised_coeffs = [pywt.threshold(c, uthresh, mode=thresholding) for c in coeffs]
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    
    # Truncate to original length if needed
    return denoised_signal[:len(noisy_signal)]


# ===========================
# 4. Generate ECG and add noise
# ===========================
ecg = generate_ecg()
noisy_ecg = add_noise(ecg)


# ===========================
# 5. Denoise using wavelet transform
# ===========================
clean_ecg = wavelet_denoising(noisy_ecg)

# ===========================
# 6. Plot original, noisy, and denoised signals
# ===========================
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(ecg, label='Original ECG')
plt.title('Original ECG Signal')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(noisy_ecg, label='Noisy ECG')
plt.title('Noisy ECG Signal')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(clean_ecg, label='Denoised ECG')
plt.title('Denoised ECG Signal using Wavelet Transform')
plt.legend()

plt.tight_layout()
plt.show()

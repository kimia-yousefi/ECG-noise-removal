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


# تابع برای حذف نویز با استفاده از تبدیل موجک
def wavelet_denoising(noisy_signal, wavelet='db4', level=4, thresholding='soft'):
    coeffs = pywt.wavedec(noisy_signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(noisy_signal)))
    denoised_coeffs = map(lambda x: pywt.threshold(x, uthresh, mode=thresholding), coeffs)
    denoised_signal = pywt.waverec(list(denoised_coeffs), wavelet)
    return denoised_signal

# تولید سیگنال ECG و افزودن نویز
ecg = generate_ecg()
noisy_ecg = add_noise(ecg)

# حذف نویز با استفاده از تبدیل موجک
clean_ecg = wavelet_denoising(noisy_ecg)

# نمایش سیگنال‌ها
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(ecg)
plt.title('Original ECG Signal')
plt.subplot(3, 1, 2)
plt.plot(noisy_ecg)
plt.title('Noisy ECG Signal')
plt.subplot(3, 1, 3)
plt.plot(clean_ecg)
plt.title('Denoised ECG Signal')
plt.tight_layout()
plt.show()



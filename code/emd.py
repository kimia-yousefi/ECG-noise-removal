import wfdb
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD, EEMD 

# ===========================
# 1. Load ECG signal (MIT-BIH record 100)
# ===========================
record = wfdb.rdrecord('100', pb_dir='mitdb')  
signal = record.p_signal[:, 0]  # select lead I


# ===========================
# 2. Normalize the signal
# ===========================
signal = (signal - np.mean(signal)) / np.std(signal)


# ===========================
# 3. Add white Gaussian noise
# ===========================
noise = np.random.normal(0, 0.1, signal.shape)
noisy_signal = signal + noise



# ===========================
# 4. Define EMD denoising function
# ===========================
def denoise_signal_with_emd(noisy_sig, exclude_first_n=1):
    """
    Apply Empirical Mode Decomposition (EMD) to denoise a signal.
    
    Parameters:
        noisy_sig: 1D numpy array of noisy signal
        exclude_first_n: number of first IMFs to exclude (usually contain high-frequency noise)
    
    Returns:
        denoised signal
    """
    emd = EMD()
    imfs = emd(noisy_sig)
    # Reconstruct signal excluding first few IMFs
    denoised = np.sum(imfs[exclude_first_n:], axis=0)
    return denoised

# ===========================
# 5. Define EEMD denoising function
# ===========================
def denoise_signal_with_eemd(noisy_sig, exclude_first_n=1):
    """
    Apply Ensemble EMD (EEMD) to denoise a signal.
    
    Parameters:
        noisy_sig: 1D numpy array of noisy signal
        exclude_first_n: number of first IMFs to exclude
    
    Returns:
        denoised signal
    """
    eemd = EEMD()
    imfs = eemd.eemd(noisy_sig)
    denoised = np.sum(imfs[exclude_first_n:], axis=0)
    return denoised

# ===========================
# 6. Apply EMD and EEMD denoising
# ===========================
denoised_signal_emd = denoise_signal_with_emd(noisy_signal)
denoised_signal_eemd = denoise_signal_with_eemd(noisy_signal)

# ===========================
# 7. Extract removed noise (optional)
# ===========================
removed_noise_emd = noisy_signal - denoised_signal_emd
removed_noise_eemd = noisy_signal - denoised_signal_eemd


# رسم سیگنال‌ها
plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
plt.plot(signal, label='Original ECG')
plt.title('Original ECG Signal')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(denoised_signal_emd, label='Denoised ECG (EMD)')
plt.title('Denoised ECG Signal using EMD')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(denoised_signal, label='Denoised ECG (EEMD)')
plt.title('Denoised ECG Signal using EEMD')
plt.legend()

plt.tight_layout()
plt.show()
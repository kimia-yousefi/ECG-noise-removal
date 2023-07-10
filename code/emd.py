import wfdb
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD, EEMD 

# ===========================
# 1. Load ECG signal (MIT-BIH record 100)
# ===========================
record = wfdb.rdrecord('100', pb_dir='mitdb')  
signal = record.p_signal[:, 0]  # select lead I


# استفاده از EMD برای حذف نویز 
def denoise_signal_with_emd(noisy_signal):
     emd = EMD() 
     imfs = emd(noisy_signal) 
     return imfs

# اضافه کردن نویز گوسی سفید به سیگنال)
noise = np.random.normal(0, 0.1, signal.shape)
noisy_signal = signal + noise


# استفاده از EEMD برای حذف نویز
def denoise_signal_with_eemd(noisy_signal):
    eemd = EEMD()
    imfs = eemd.eemd(noisy_signal)
    return imfs

# بازسازی سیگنال از IMFs انتخابی
def reconstruct_signal(imfs, exclude_first_n=1):
    return np.sum(imfs[exclude_first_n:], axis=0)

# اعمال EMD
imfs_emd = denoise_signal_with_emd(noisy_signal)
denoised_signal_emd = reconstruct_signal(imfs_emd)

# اعمال EEMD
imfs = denoise_signal_with_eemd(noisy_signal)

# بازسازی سیگنال تمیز شده
denoised_signal = reconstruct_signal(imfs)

# استخراج نویز حذف شده
removed_noise = noisy_signal - denoised_signal

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
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, UpSampling1D

# ===========================
# 1. Load ECG signal (MIT-BIH record 100)
# ===========================
record = wfdb.rdrecord('100', pb_dir='mitdb')
signal = record.p_signal[:, 0]

# ===========================
# 2. Normalize the signal
# ===========================
signal = (signal - np.mean(signal)) / np.std(signal)


# ===========================
# 3. Add noise (Base Wandering + Muscle Artifact)
# ===========================
t = np.arange(len(signal)) / record.fs
bw_noise = 0.5 * np.sin(0.1 * np.pi * t)
ma_noise = 0.1 * np.random.normal(0, 1, signal.shape)
noisy_signal = signal + bw_noise + ma_noise

# افزودن نویز Muscle Artifact (MA)
ma_noise = 0.1 * np.random.normal(0, 1, signal.shape)

# ترکیب نویزها با سیگنال اصلی
noisy_signal = signal + bw_noise + ma_noise

# تقسیم داده‌ها به بخش‌های آموزشی و تست
train_size = int(len(signal) * 0.8)
train_signal = noisy_signal[:train_size]
test_signal = noisy_signal[train_size:]

# تعریف مدل FCN-based DAE
input_signal = Input(shape=(None, 1))

# لایه‌های کانولوشن
x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(input_signal)
x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)

# لایه‌های Upsampling
x = UpSampling1D(size=2)(x)
x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
x = Conv1D(1, kernel_size=3, activation=None, padding='same')(x)

model = Model(input_signal, x)
model.compile(optimizer='adam', loss='mse')

# آماده‌سازی داده‌ها برای آموزش
train_signal = train_signal.reshape(-1, 1)
test_signal = test_signal.reshape(-1, 1)

# آموزش مدل
model.fit(train_signal, train_signal, epochs=10, batch_size=32, validation_data=(test_signal, test_signal))

# استفاده از مدل برای حذف نویز
denoised_signal = model.predict(noisy_signal.reshape(-1, 1)).flatten()

# رسم سیگنال‌ها
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(signal, label='Original ECG')
plt.title('Original ECG Signal')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(noisy_signal, label='Noisy ECG (BW + MA)')
plt.title('Noisy ECG Signal with BW and MA')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(denoised_signal, label='Denoised ECG (FCN-based DAE)')
plt.title('Denoised ECG Signal using FCN-based DAE')
plt.legend()

plt.tight_layout()
plt.show()
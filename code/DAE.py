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


# ===========================
# 4. Segment the signal for training
# ===========================
def create_segments(sig, seg_len=256):
    """Divide 1D signal into non-overlapping segments for training."""
    segments = []
    for i in range(0, len(sig) - seg_len, seg_len):
        segments.append(sig[i:i+seg_len])
    return np.array(segments)[..., np.newaxis]  # shape: (num_segments, seg_len, 1)

train_size = int(len(signal) * 0.8)
train_segments = create_segments(noisy_signal[:train_size])
test_segments = create_segments(noisy_signal[train_size:])


# ===========================
# 5. Build FCN-based DAE model
# ===========================
input_signal = Input(shape=(train_segments.shape[1], 1))  # fixed segment length

x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(input_signal)
x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
x = Conv1D(1, kernel_size=3, activation=None, padding='same')(x)  # output same shape

model = Model(input_signal, x)
model.compile(optimizer='adam', loss='mse')


# ===========================
# 6. Train the model
# ===========================
model.fit(
    train_segments, train_segments,
    validation_data=(test_segments, test_segments),
    epochs=20,
    batch_size=32
)


# ===========================
# 7. Denoise full noisy signal
# ===========================
# Segment the full noisy signal
all_segments = create_segments(noisy_signal)
denoised_segments = model.predict(all_segments)

# Reconstruct full denoised signal
denoised_signal = denoised_segments.reshape(-1)[:len(noisy_signal)]  # truncate to original length


# ===========================
# 8. Plot signals
# ===========================
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(signal, label='Original ECG')
plt.title('Original ECG Signal')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(noisy_signal, label='Noisy ECG (BW + MA)')
plt.title('Noisy ECG Signal with Base Wandering and Muscle Artifact')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(denoised_signal, label='Denoised ECG (FCN-based DAE)')
plt.title('Denoised ECG Signal using FCN-based DAE')
plt.legend()

plt.tight_layout()
plt.show()
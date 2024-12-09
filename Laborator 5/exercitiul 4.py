import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

# Setul de date
(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()

# Normalizarea
X_train = X_train / 255.0
X_test = X_test / 255.0

# Adaugam canalul dimension ca sa mearga pe Conv2D
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Generam anomalii (introducem zgomot)
noise_factor = 0.35
X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
X_test_noisy = np.clip(X_test_noisy, 0.0, 1.0)  # Keep pixel values in [0, 1]


# Autoencoderul
class ConvAutoencoder(Model):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2, input_shape=(28, 28, 1)),
            layers.Conv2D(4, (3, 3), activation='relu', padding='same', strides=2)
        ])

        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(4, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2DTranspose(8, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')  # Output layer
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


# Initializam modelul
autoencoder = ConvAutoencoder()
autoencoder.compile(optimizer='adam', loss='mse')

# Antrenam autoencoderul
history = autoencoder.fit(
    X_train, X_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, X_test),
    verbose=1
)

# Calculam reconstruction loss pentru training data
reconstructed_train = autoencoder(X_train)
train_errors = np.mean(np.square(X_train - reconstructed_train), axis=(1, 2, 3))

# Calculam thresholdul
threshold = np.mean(train_errors) + np.std(train_errors)

# Calculam reconstruction loss pentru test data si noisy test data
reconstructed_test = autoencoder(X_test)
reconstructed_test_noisy = autoencoder(X_test_noisy)
test_errors = np.mean(np.square(X_test - reconstructed_test), axis=(1, 2, 3))
noisy_errors = np.mean(np.square(X_test_noisy - reconstructed_test_noisy), axis=(1, 2, 3))


# Clasificam anomaliile pe baza threshold-ului
y_train_pred = (train_errors > threshold).astype(int)
y_test_pred = (test_errors > threshold).astype(int)
y_noisy_pred = (noisy_errors > threshold).astype(int)

print(f"Threshold for anomaly detection: {threshold:.4f}")
print(f"Accuracy on clean test images: {1 - y_test_pred.mean():.4f}")
print(f"Accuracy on noisy test images: {1 - y_noisy_pred.mean():.4f}")

# Afisam rezultatele
n = 5  # cate numere afisam

plt.figure(figsize=(10, 8))

for i in range(n):
    # Imaginile originale
    plt.subplot(4, n, i + 1)
    plt.imshow(X_test[i].squeeze(), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Imaginile cu zgomot
    plt.subplot(4, n, i + 1 + n)
    plt.imshow(X_test_noisy[i].squeeze(), cmap='gray')
    plt.title("Noisy")
    plt.axis('off')

    # Imaginile reconstruite din alea originale
    plt.subplot(4, n, i + 1 + 2 * n)
    plt.imshow(reconstructed_test[i].numpy().squeeze(), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

    # Imaginile reconstruite din alea noisy
    plt.subplot(4, n, i + 1 + 3 * n)
    plt.imshow(reconstructed_test_noisy[i].numpy().squeeze(), cmap='gray')
    plt.title("Denoised")
    plt.axis('off')

plt.tight_layout()
plt.show()

# Denoiser in efect
history_denoise = autoencoder.fit(
    X_test_noisy, X_test,  # input: imaginile noisy, ce vrem: imagini denoised
    epochs=10,
    batch_size=64,
    verbose=1
)

# Rezultatele denoiserului
reconstructed_denoised = autoencoder(X_test_noisy)

plt.figure(figsize=(10, 8))

for i in range(n):
    # Imaginile noisy
    plt.subplot(3, n, i + 1)
    plt.imshow(X_test_noisy[i].squeeze(), cmap='gray')
    plt.title("Noisy")
    plt.axis('off')

    # Imaginile denoised
    plt.subplot(3, n, i + 1 + n)
    plt.imshow(reconstructed_denoised[i].numpy().squeeze(), cmap='gray')
    plt.title("Denoised")
    plt.axis('off')

    # Imaginile originale
    plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(X_test[i].squeeze(), cmap='gray')
    plt.title("Original")
    plt.axis('off')

plt.tight_layout()
plt.show()
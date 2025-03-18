import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as signal
import os

# Cargar el archivo de audio
archivo_audio = "Archivo.wav"  # Reemplázalo con el archivo de tu cancion, preferiblemente en formato WAV
print(os.path.exists(archivo_audio))  # Debería retornar True si el archivo está ahí
frecuencia_muestreo, audio = wav.read(archivo_audio)

# Si el audio es estéreo, convertiertelo a mono para una mayor facilidad de analisis
if len(audio.shape) > 1:
    audio = np.mean(audio, axis=1)

# Tomar solo 200 segundos para el análisis, puedes modificar la cantidad a tu gusto, dependiendo de tu potencia de hardware
duracion_muestras = frecuencia_muestreo * 200  # 200 segundos
audio = audio[:duracion_muestras]

# Generar el espectrograma
frecuencia, tiempos, Sxx = signal.spectrogram(audio, fs=frecuencia_muestreo, nperseg=1024)
Sxx = np.maximum(Sxx, 1e-10) 
espectrograma_dB = 10 * np.log10(Sxx)

# Cálculo de autocorrelación optimizado con FFT
audio -= np.mean(audio)  # Eliminar componente DC
autocorr = np.fft.ifft(np.abs(np.fft.fft(audio))**2).real
autocorr = autocorr[:len(autocorr) // 2]  # Solo mitad positiva
autocorr /= np.max(autocorr)  # Normalizar

# Detección de picos en la autocorrelación (posibles ondas estacionarias)
picos, _ = signal.find_peaks(autocorr, height=0.2, distance=frecuencia_muestreo//2)

# Gráficos
plt.figure(figsize=(12, 8))

# Espectrograma
plt.subplot(2, 1, 1)
plt.pcolormesh(tiempos, frecuencia, espectrograma_dB, shading='auto', cmap='inferno')
plt.colorbar(label="Amplitud (dB)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Frecuencia (Hz)")
plt.title("Espectrograma de la Canción")

# Autocorrelación
plt.subplot(2, 1, 2)
plt.plot(autocorr, label="Autocorrelación de la Señal", color='blue')
plt.scatter(picos, autocorr[picos], color='red', label="Patrones Repetitivos (Picos)")
plt.xlabel("Retraso (muestras)")
plt.ylabel("Amplitud Normalizada")
plt.title("Detección de Ondas Estacionarias en la Señal")
plt.legend()

plt.tight_layout()
plt.show()

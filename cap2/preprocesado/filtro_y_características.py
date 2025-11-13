"""
paso 1:
1er paso del preprocesado:
PREPARACIÓN DATOS PREVIO AL CLUSTERING - FILTRO DE CALIDAD Y EXTRACCIÓN DE CARACTERÍSTICAS
"""
import os
import gzip
import pickle
import numpy as np
from scipy.stats import skew
from scipy.signal import find_peaks
from joblib import Parallel, delayed

# Parámetros 
fs = 250
ventana_seg = 25 # duración ventana
ventana_samples = fs * ventana_seg

ruta_disco = "D:/descargasTFG/icentia11k"
salida_dir = "D:/descargasTFG/Icentia11k_features"
os.makedirs(salida_dir, exist_ok=True)

log_file = os.path.join(salida_dir, "ventanas_validas_log.txt")


def calidad_ventanas(ecg_ventana, fs = 250, std_min = 0.02, max_z = 8, prominencia = 0.3):
    
    if np.std(ecg_ventana) < std_min:
        return None, None
    ventana_z = (ecg_ventana - np.mean(ecg_ventana)) / (np.std(ecg_ventana) + 1e-8)
    if np.max(ventana_z) > max_z or np.min(ventana_z) < -max_z:
        return None, None
    peaks, _ = find_peaks(ventana_z, distance=int(0.3*fs), prominence = prominencia)
    etiqueta = "valida" if len(peaks) >= 2 else "rara"
    return etiqueta, ventana_z

def extraer_caracteristicas(ecg_ventana):
    
    features = []
    subsegmentos = np.array_split(ecg_ventana, 3)
    for seg in subsegmentos:
        features.append(np.mean(seg))
        features.append(np.std(seg))
        features.append(skew(seg))
    fft_vals = np.abs(np.fft.fft(ecg_ventana))
    n = len(ecg_ventana)
    freq = np.fft.fftfreq(n, 1/fs)
    lf = np.sum(fft_vals[(freq >= 0.04) & (freq < 0.15)])
    hf = np.sum(fft_vals[(freq >= 0.15) & (freq < 0.4)])
    features.extend([lf, hf])
    peaks, _ = find_peaks(ecg_ventana, distance=0.6*fs)
    rr_intervals = np.diff(peaks)/fs
    if len(rr_intervals) > 0:
        features.append(np.mean(rr_intervals))
        features.append(np.std(rr_intervals))
    else:
        features.extend([0, 0])
    features.append(np.max(ecg_ventana))
    features.append(np.min(ecg_ventana))
    return np.array(features)

def features_con_etiqueta(ecg_ventana):
    
    etiqueta, ventana_z = calidad_ventanas(ecg_ventana)
    if ventana_z is None:
        return None
    features = extraer_caracteristicas(ventana_z)
    return np.append(features, 0 if etiqueta == "valida" else 1)

def procesar_paciente(paciente, fs=250):
    
    features_all = []

    # verificar que paciente sea un array válido
    if not isinstance(paciente, (np.ndarray, list)):
        return None

    paciente = np.array(paciente)
    if paciente.size < ventana_samples:
        return None  # señal demasiado corta

    # ventanas de 25 segundos
    for start in range(0, len(paciente) - ventana_samples + 1, ventana_samples):
        ventana = paciente[start:start + ventana_samples]
        feats = features_con_etiqueta(ventana)
        if feats is not None:
            features_all.append(feats)

    if len(features_all) == 0:
        return None

    return np.vstack(features_all)

def procesar_archivo(ruta_archivo):
    
    print("Procesando:", ruta_archivo)
    with gzip.open(ruta_archivo, 'rb') as f:
        datos = pickle.load(f)

    # procesar pacientes en paralelo
    todas_features = Parallel(n_jobs=-1)(
        delayed(procesar_paciente)(paciente, fs=fs) for paciente in datos
    )

    arrays_validos = [x for x in todas_features if x is not None]
    n_ventanas = sum(x.shape[0] for x in arrays_validos) if arrays_validos else 0

    if len(arrays_validos) == 0:
        print("ninguna ventana válida en este archivo:", ruta_archivo)
        return None, n_ventanas

    todas_features = np.vstack(arrays_validos)
    print(f"ventanas válidas en este archivo: {n_ventanas}")
    return todas_features, n_ventanas

# Procesar todos los archivos 
archivos = [f for f in os.listdir(ruta_disco) if f.endswith('_batched.pkl.gz')]
archivos = [os.path.join(ruta_disco, f) for f in archivos]

# Abrir log
with open(log_file, "w") as log:
    log.write("archivo\tVventanas_validas\n")

for archivo in archivos:
    feats, n_ventanas = procesar_archivo(archivo)
    with open(log_file, "a") as log:
        log.write(f"{os.path.basename(archivo)}\t{n_ventanas}\n")
    if feats is not None:
        nombre = os.path.basename(archivo).replace(".pkl.gz", "_features.pkl")
        salida_archivo = os.path.join(salida_dir, nombre)
        with open(salida_archivo, "wb") as f:
            pickle.dump(feats, f)
        print(f"guardado: {salida_archivo}")

print(f"log guardado en: {log_file}")


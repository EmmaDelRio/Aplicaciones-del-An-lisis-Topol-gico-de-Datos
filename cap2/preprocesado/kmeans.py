"""
paso 3:
ALGORIMTO CLUSTER K-MEANS
"""

import os, pickle, zlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

def load_trabajos(asignacion_file, col_arch=0, col_k=3):
    # carga el .csv donde tenemos la info de los k
    
    df = pd.read_csv(asignacion_file, usecols=[col_arch, col_k]).dropna()
    df.iloc[:, 0] = df.iloc[:, 0].str.replace('_batched.pkl.gz', '', regex=False)
    df.iloc[:, 1] = df.iloc[:, 1].astype(int)
    return list(df.itertuples(index=False, name=None))

# aseguramos reproducibilidad
def seed_from(name):
    
    return zlib.adler32(name.encode("utf-8")) & 0xFFFFFFFF

def medoids_from_labels(X_std, centers, labels, k):
    
    dists = cdist(X_std, centers, metric="sqeuclidean")
    idx = []
    for j in range(k):
        mask = np.where(labels == j)[0]
        if mask.size > 0:
            local = mask[np.argmin(dists[mask, j])]
            idx.append(local)
        else:
            idx.append(np.random.randint(0, len(X_std)))
    return np.array(idx, dtype=np.int64)

def procesar_archivo(features_dir, salida_dir, archivo_base, k):
    
    ruta_in = features_dir / f"{archivo_base}_batched_features.pkl"
    ruta_repr = salida_dir / f"{archivo_base}_repr.pkl"
    ruta_idx = salida_dir / f"{archivo_base}_repr_idx.npy"

    if not ruta_in.exists():
        print(f" saltado: {archivo_base} (no encontrado)")
        return

    try:
        with open(ruta_in, "rb") as f:
            X = np.array(pickle.load(f), dtype=np.float32)  # (n,16)

        # separar features (15) y etiqueta (1)
        X_feats = X[:, :15]
        y = X[:, 15].astype(np.int32)

        # quedarnos solo con válidas
        mask_validas = (y == 0)
        X_feats_validas = X_feats[mask_validas]

        n, d = X_feats_validas.shape
        k = int(k)
        seed = seed_from(archivo_base)

        if n == 0:
            print(f"{archivo_base}: 0 válidas (saltado)")
            return

        if n <= k:
            reps, idx = X_feats_validas, np.arange(n)
            print(f"{archivo_base}: {n} válidas (caso trivial)")
        else:
            # estandarizamos para que todas las features cuenten igual
            scaler = StandardScaler()
            X_std = scaler.fit_transform(X_feats_validas)

            model = KMeans(n_clusters=k, random_state=seed, n_init="auto")
            labels = model.fit_predict(X_std)
            idx = medoids_from_labels(X_std, model.cluster_centers_, labels, k)
            reps = X_feats_validas[idx]
            print(f"{archivo_base}")

        # guardar representantes (15 features, sin etiqueta) e índices relativos al conjunto filtrado
        with open(ruta_repr, "wb") as f:
            pickle.dump(reps, f)
        np.save(ruta_idx, idx)

    except Exception as e:
        print(f" error en {archivo_base}: {str(e)}")

def main():
    features_dir = Path("D:/descargasTFG/Icentia11k_features")
    asignacion_file = features_dir / "asignacion_k.csv"
    salida_dir = features_dir / "representantes1"
    salida_dir.mkdir(exist_ok=True)

    trabajos = load_trabajos(asignacion_file, col_k=3)
    print(f"encontrados {len(trabajos)} pacientes a procesar")

    for i, (archivo_base, k) in enumerate(trabajos):
        print(f"procesando {i+1}/{len(trabajos)}: {archivo_base} (k={k})")
        procesar_archivo(features_dir, salida_dir, archivo_base, k)

    print("fin")

if __name__ == "__main__":
    main()

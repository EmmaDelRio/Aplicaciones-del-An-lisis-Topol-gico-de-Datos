"""
paso 4:
CREACIÓN DEL CONJUNTO DE DATOS DE ENTRADA A MAPPER
"""
import numpy as np
import pickle
from pathlib import Path
import json

def crear_dataset_completo(features_dir, output_dir="mapper_dataset"):
    # dataset final con representantes (15 features) + info para rastrear al índice original
    
    features_dir = Path(features_dir)
    representantes_dir = features_dir / "1representantes"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"buscando en: {representantes_dir}")

    archivos_repr = list(representantes_dir.glob("*_repr.pkl"))
    print(f"encontrados {len(archivos_repr)} archivos _repr.pkl")

    todos_representantes = []
    metadata_detallada = []

    for i, archivo_repr in enumerate(archivos_repr):
        try:
            paciente_id = archivo_repr.stem.replace("_repr", "")

            if i % 500 == 0:
                print(f"procesando {i}/{len(archivos_repr)}: {paciente_id}...")

            # cargar representantes e índices relativos al conjunto filtrado
            representantes = pickle.load(open(archivo_repr, "rb"))          # (k, 15)
            indices_filtrado = np.load(representantes_dir / f"{paciente_id}_repr_idx.npy")  # idx en filtrado

            # cargar datos originales (16 cols)
            archivo_original = features_dir / f"{paciente_id}_batched_features.pkl"
            datos_originales = np.array(pickle.load(open(archivo_original, "rb")), dtype=np.float32)

            # reconstruir máscara de válidas y mapeo a índice original
            X_feats = datos_originales[:, :15]
            y = datos_originales[:, 15].astype(np.int32)
            mask_validas = (y == 0)
            idx_a_original = np.where(mask_validas)[0]   # posición en X original de cada fila válida

            #  índices de representantes  -> índices originales
            indices_original = idx_a_original[indices_filtrado]

            # verificación
            for j, (repr_point, idx_orig) in enumerate(zip(representantes, indices_original)):
               
                iguales = np.allclose(repr_point, X_feats[idx_orig], atol=1e-6)
                if not iguales:
                    print(f"no coincide {paciente_id}, cluster {j} (idx_orig={int(idx_orig)})")

                metadata_detallada.append({
                    "paciente_id": paciente_id,
                    "cluster_idx": int(j),
                    "indice_original": int(idx_orig),
                    "archivo_original": archivo_original.name
                })

                todos_representantes.append(repr_point)

        except Exception as e:
            print(f"error con {paciente_id}: {e}")
            continue

    # convertir a array
    X_mapper = np.array(todos_representantes, dtype=np.float32)  # (n_total, 15)

    # guardar
    np.save(output_dir / "X_mapper.npy", X_mapper)

    with open(output_dir / "metadata_detallada.json", "w") as f:
        json.dump(metadata_detallada, f, indent=2)

    # paciente -> índices originales
    mapping_pacientes = {}
    for item in metadata_detallada:
        mapping_pacientes.setdefault(item["paciente_id"], []).append(item["indice_original"])

    with open(output_dir / "mapping_pacientes.json", "w") as f:
        json.dump(mapping_pacientes, f, indent=2)

    print("\n Dataset creado:")
    print(f"   - Puntos totales: {X_mapper.shape[0]}")
    print(f"   - Dimensiones: {X_mapper.shape[1]}")
    print(f"   - Pacientes únicos: {len(mapping_pacientes)}")
    print(f"   - Archivos de salida guardados en: {output_dir}")

    return X_mapper, metadata_detallada

if __name__ == "__main__":
    features_dir = "D:/descargasTFG/Icentia11k_features"
    output_dir = r"C:\Users\Emma\Desktop\cap2" 
    
    print("inicio")
    print(f"carpeta de características: {features_dir}")

    X_mapper, metadata = crear_dataset_completo(features_dir)

    if X_mapper is not None:
        print("\nfin")
    else:
        print("\nproceso fallido")

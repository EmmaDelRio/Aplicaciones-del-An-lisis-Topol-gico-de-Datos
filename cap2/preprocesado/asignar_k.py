"""
paso 2:
ASIGNACIÓN DEL NÚMERO DE REPRESENTANTES POR PACIENTE
    proporcional al número de ventanas disponibles
"""

import pandas as pd
import numpy as np

# lee log 
log_file = "D:/descargasTFG/Icentia11k_features/ventanas_validas_log.txt"
df = pd.read_csv(log_file, sep="\t")
df.columns = ["Archivo","Ventanas_validas"]

total_ventanas = df["Ventanas_validas"].sum()
n_pacientes = df.shape[0]

# numero total de datos y min,max por paciente
D = 100000   
min_k = 3
max_k = 50

# asignación proporcional
df["k_prop"] = (df["Ventanas_validas"] * D / total_ventanas).round().astype(int)
df["k"] = df["k_prop"].clip(lower=min_k, upper=max_k)

# ajuste para aproximar suma a D 
suma = df["k"].sum()
print("Total asignado inicialmente:", suma)

# para que la suma sea exactamene D
diff = int(D - suma)
if diff != 0:
    
    idx_order = df.sort_values("Ventanas_validas", ascending=False).index
    i = 0
    step = 1 if diff>0 else -1
    while diff != 0 and i < len(idx_order):
        idx = idx_order[i]
        new_val = df.at[idx, "k"] + step
        if min_k <= new_val <= max_k:
            df.at[idx, "k"] = new_val
            diff -= step
        i += 1
        if i == len(idx_order) and diff != 0:
            i = 0  

print("Total asignado tras ajuste:", df["k"].sum())

# vemos algunos ejemplos
print(df[["Archivo","Ventanas_validas","k"]].head())

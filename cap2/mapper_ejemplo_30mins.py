"""
EJEMPLO INICIAL APLICACIÓN ALGORITMO MAPPER 
    se aplica el algoritmo a 30  mins aleatorios de un único paciente
"""

import os
import gzip
import pickle
import numpy as np
from scipy.signal import find_peaks, welch
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo
import matplotlib.pyplot as plt
import kmapper as km
import random
import networkx as nx

# CONFIGURACIÓN BÁSICA

fs = 250          # frecuencia de muestreo
ventana_seg = 60  # duración de ventana que se desee
ruta_disco = "D:/descargasTFG/icentia11k"  
salida = "D:/descargasTFG/ejemplo1"  
os.makedirs(salida, exist_ok = True)

# parámetros para el Mapper
EPS = 0.6
MIN_SAMPLES = 4
N_CUBES = 6
OVERLAP = 0.25

def extraer_caracteristicas_simples(ventana_ecg, fs = 250):
    # Extrae 4 características básicas
    
    ventana_ecg = np.nan_to_num(ventana_ecg, nan = 0.0) # manejo de NaN
    
    if np.std(ventana_ecg) < 1e-6: # señal plana -> valores por defecto
        return np.array([60.0, 0.05, 0.1, 0.1])
    
    try:
        # Frecuencia cardíaca
        altura_umbral = np.percentile(np.abs(ventana_ecg), 70) * 0.5
        picos, _ = find_peaks(ventana_ecg, distance  = int(0.4*fs), height = altura_umbral)
        
        if len(picos) > 2:
            rr_intervals = np.diff(picos) / fs
            hr = 60 / np.mean(rr_intervals)
            rr_std = np.std(rr_intervals)
        else:
            hr = 60.0
            rr_std = 0.05
        
        # Amplitud
        amp_picos = np.mean(np.abs(ventana_ecg[picos])) if len(picos) > 0 else np.std(ventana_ecg)
        
        # Potencia LF
        try:
            f, Pxx = welch(ventana_ecg, fs = fs, nperseg = min(256, len(ventana_ecg)))
            mask_lf = (f >= 0.04) & (f < 0.15)
            lf_power = np.mean(Pxx[mask_lf]) if np.any(mask_lf) else 0.1
        except:
            lf_power = 0.1
        
        return np.array([hr, rr_std, amp_picos, lf_power])
        
    except:
        return np.array([60.0, 0.05, 0.1, 0.1])

def aplicar_mapper(features):
    # Aplica algorimto Mapper
    
    features = np.nan_to_num(features, nan = 0.0)
    features_norm = StandardScaler().fit_transform(features)
    
    # Función filtro: proyección a 2D para visualización
    projected_data = PCA(n_components=2).fit_transform(features_norm)
    
    # Mapper 
    mapper = km.KeplerMapper()
    graph = mapper.map(projected_data,
                      features_norm,
                      clusterer=DBSCAN(eps = EPS, min_samples = MIN_SAMPLES),
                      cover=km.Cover(n_cubes = N_CUBES, perc_overlap = OVERLAP))
    
    return graph

def guardar_grafo(graph, paciente_id):
    
    plt.figure(figsize = (12, 10))
    
    # Convierte a grafo de networkx para mejor control
    G = nx.Graph()
    
    # Añade nodos con sus tamaños
    node_sizes = []
    for node_id, indices in graph['nodes'].items():
        size = len(indices)
        G.add_node(node_id, size = size)
        node_sizes.append(size * 100)  # escalar
    
    # Añade aristas
    for node_id, connections in graph['links'].items():
        for connected_node in connections:
            G.add_edge(node_id, connected_node)
    
    # Layout para separar nodos
    pos = nx.spring_layout(G, k = 2, iterations = 50)
    
    # Dibuja nodos
    nx.draw_networkx_nodes(G, pos, 
                          node_size = node_sizes,
                          node_color='lightblue',
                          alpha = 0.8,
                          edgecolors = 'black',
                          linewidths = 2)
    
    # Dibuja aristas 
    nx.draw_networkx_edges(G, pos,
                          edge_color = 'gray',
                          width = 2,
                          alpha = 0.6)
    
    plt.axis('off')  # Ocultar ejes
    plt.grid(False)
    
    img_path = os.path.join(salida, f"{paciente_id}_grafo_mejorado.png")
    plt.savefig(img_path, dpi = 300, bbox_inches = 'tight', facecolor = 'white')
    plt.close()
    
    print(f"Grafo mejorado guardado: {img_path}")
    return img_path

def mostrar_info_grafo(graph):
    # Muestra información básica del grafo
    
    print("\n info del grafo")
    print(f"\n nodos: {len(graph['nodes'])}")
    print(f"\n aristas: {sum(len(v) for v in graph['links'].values())}")
    
    tamanos = [len(nodo) for nodo in graph['nodes'].values()]
    if tamanos:
        print(f"\n tamaño máximo: {max(tamanos)}")
        print(f"\n tamaño mínimo: {min(tamanos)}")
        print(f"\n promedio: {np.mean(tamanos):.1f}")

def main():
    print("ejemplo básico Mapper")
    
    # Fijar semilla
    random.seed(13)  
    np.random.seed(13)
    
    # Busca archivo
    archivos = [f for f in os.listdir(ruta_disco) if f.endswith(".pkl.gz")]
    
    if not archivos:
        print(" no he encontrado archivos")
        return
    
    archivo = archivos[0]
    ruta_completa = os.path.join(ruta_disco, archivo)
    
    print(f"\n procesando: {archivo}")
    print("\n cargando datos...")
    
    try:
        with gzip.open(ruta_completa, "rb") as f:
            data = pickle.load(f)
        
        # Toma primer paciente y 30 minutos de datos
        señal = data[0]
        duracion = 1800 * fs  
        inicio = random.randint(0, len(señal) - duracion)
        segmento = señal[inicio:inicio + duracion]
        
        print(f"\n segmento: {len(segmento)/fs/60:.1f} minutos")
        
        # Extrae características
        ventana_muestras = ventana_seg * fs
        num_ventanas = len(segmento) // ventana_muestras
        caracteristicas = []
        
        print(f"\n procesando {num_ventanas} ventanas")
        
        for i in range(num_ventanas):
            ventana = segmento[i*ventana_muestras:(i+1)*ventana_muestras]
            carac = extraer_caracteristicas_simples(ventana, fs)
            caracteristicas.append(carac)
        
        caracteristicas = np.array(caracteristicas)
        print(f"\n características extraídas: {caracteristicas.shape}")
        
        # Aplicar Mapper
        grafo = aplicar_mapper(caracteristicas)
        
        # Mostrar información
        mostrar_info_grafo(grafo)
        
        # Guardar grafo
        guardar_grafo(grafo, "ejemplo_mapper")
        
        print("\n fin")
        
        
    except Exception as e:
        print(f"error: {e}")

if __name__ == "__main__":
    main()
"""
MAPPER CONFIGURACIÓN B
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
import umap
import hdbscan
import kmapper as km
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

# Entradas

RUTA_X = Path("X_mapper.npy")
assert RUTA_X.exists(), "No se encuentra X_mapper.npy"

# nombres  características
nombres_caracteristicas = [
    "seg1_media", "seg1_desv", "seg1_asim",
    "seg2_media", "seg2_desv", "seg2_asim",
    "seg3_media", "seg3_desv", "seg3_asim",
    "LF", "HF", "RR_media", "RR_desv", "amp_max", "amp_min"
]

X = np.load(RUTA_X)
if X.ndim != 2:
    raise ValueError("X debe ser 2D")
assert X.shape[1] == 15, f"Se esperaban 15 columnas, se obtuvo {X.shape[1]}"

#Previo

escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)

pca = PCA(n_components = 10, random_state = 42)
X_pca = pca.fit_transform(X_escalado)
print("Varianza explicada acumulada:", float(pca.explained_variance_ratio_.cumsum()[-1]))

# UMAP 2D (visualización solo)

umap_2d = umap.UMAP(
    n_neighbors = 30,
    min_dist = 0.05,
    n_components = 2,
    metric = "euclidean",
    random_state = 42
).fit_transform(X_pca)

# Función filtro: excenricidad 1D

def excentricidad_knn(X_emb, k = 50):
    vecinos = NearestNeighbors(n_neighbors = k+1, metric = "euclidean", n_jobs = 1).fit(X_emb)
    dist, _ = vecinos.kneighbors(X_emb)
    dist = dist[:, 1:]
    return dist.mean(axis = 1)

k_ecc = 30
ecc_1d = excentricidad_knn(X_pca, k=k_ecc)
ecc_z = (ecc_1d - np.mean(ecc_1d)) / (np.std(ecc_1d) + 1e-8)

def grafo_a_networkx(dic_grafo):

    G = nx.Graph()
    
    for id_nodo, indices_puntos in dic_grafo["nodes"].items():
        G.add_node(id_nodo, tamano=len(indices_puntos))
    
    for origen, destino in dic_grafo.get("links", []):
        G.add_edge(origen, destino)
    
    return G

def posiciones_desde_umap(dic_grafo, umap_2d):
    pos = {}
    for id_nodo, indices in dic_grafo["nodes"].items():
        pts = umap_2d[np.array(indices)]
        pos[id_nodo] = pts.mean(axis=0)
    return pos

def dibujar_png(dic_grafo, posiciones, valores_nodo = None,
                titulo = "", salida = "out.png", vmin = None, vmax = None, tam_titulo = 16):

    G = grafo_a_networkx(dic_grafo)
    tamanos = [max(10, np.log10(G.nodes[n]["tamano"] + 1) * 60) for n in G.nodes()]

    fig, ax = plt.subplots(figsize = (10, 10))
    nx.draw_networkx_edges(G, posiciones, ax = ax, width = 1.2, edge_color = "#666666")

    if valores_nodo is None:
        nx.draw_networkx_nodes(G, posiciones, ax = ax, node_size = tamanos, node_color = "#3182bd")
    else:
        vals = np.array([valores_nodo.get(n, np.nan) for n in G.nodes()])
        finitos = vals[np.isfinite(vals)]
        if finitos.size == 0:
            vals[:] = 0.0
        else:
            relleno = np.nanmedian(finitos)
            vals = np.nan_to_num(vals, nan = relleno,
                                 posinf = finitos.max(), neginf = finitos.min())

        dib = nx.draw_networkx_nodes(
            G, posiciones, ax = ax, node_size = tamanos,
            node_color = vals, cmap = "viridis", vmin = vmin, vmax = vmax
        )

        norm = mpl.colors.Normalize(
            vmin = np.min(vals) if vmin is None else vmin,
            vmax = np.max(vals) if vmax is None else vmax
        )

        sm = mpl.cm.ScalarMappable(cmap = dib.get_cmap(), norm = norm)
        sm.set_array([])
        fig.colorbar(sm, ax = ax, shrink = 0.8).ax.set_ylabel("valor medio por nodo")

    ax.set_title(titulo, fontsize = tam_titulo)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(salida, dpi=300)
    plt.close(fig)
    print(f"PNG: {salida}")

def media_por_nodo(dic_grafo, vector):
    return {id_nodo: float(np.mean(vector[np.array(indices)]))
            for id_nodo, indices in dic_grafo["nodes"].items()}

kmapper = km.KeplerMapper(verbose = 1)
cubrimiento = km.Cover(n_cubes = 22, perc_overlap = 0.50)
clusterer = hdbscan.HDBSCAN(min_cluster_size = 8, min_samples = 4, prediction_data = True)

grafo = kmapper.map(
    ecc_z.reshape(-1, 1),
    X_pca,
    cover = cubrimiento,
    clusterer = clusterer
)

# Estadísticas

G = grafo_a_networkx(grafo)
n_nodos = G.number_of_nodes()
n_aristas = G.number_of_edges()
grados = dict(G.degree())
aislados = sum(1 for d in grados.values() if d == 0)
pct_aislados = 100.0 * aislados / n_nodos if n_nodos else 0.0
componentes = list(nx.connected_components(G))
aristas_por_nodo = n_aristas / n_nodos if n_nodos else float('nan')
mayor = max((len(c) for c in componentes), default=0)
pct_mayor = (mayor / n_nodos * 100) if n_nodos else 0.0

print(
    f"nodos: {n_nodos}  aristas: {n_aristas}  comp: {len(componentes)}  "
    f"aislados: {aislados} ({pct_aislados:.1f}%)  aristas/nodo: {aristas_por_nodo:.2f}  "
    f"componente mayor: {mayor} nodos ({pct_mayor:.1f}%)"
)

# html base

X_numerico = pd.DataFrame(X, columns=nombres_caracteristicas).values

html_base = "mapper_configuracionB.html"
kmapper.visualize(
    grafo,
    path_html=html_base,
    title="Configuración B",
    X=X_numerico, X_names=nombres_caracteristicas,
    color_values=ecc_z.reshape(-1, 1),
    color_function_name=["mean"]
)
print("HTML generado:", html_base)

# png base

posiciones = posiciones_desde_umap(grafo, umap_2d)

dibujar_png(
    grafo, posiciones,
    titulo="Configuración B",
    tam_titulo=25,
    salida="configB_base.png"
)

# métricas extra
k_knn = 30
vecinos = NearestNeighbors(n_neighbors = k_knn+1, metric = "euclidean", n_jobs = 1).fit(X_pca)
dist, _ = vecinos.kneighbors(X_pca)
knn_dist = dist[:, 1:].mean(axis = 1)

lof = LocalOutlierFactor(n_neighbors = k_knn, contamination = 'auto')
lof.fit(X_pca)
lof_score = -lof.negative_outlier_factor_

metricas_extra = {
    "Eccentricity_Z": ecc_z,
    "kNN_Dist": knn_dist,
    "LOF_score": lof_score
}

for nombre, vec in metricas_extra.items():

    html = f"mapper_configB_{nombre}.html"
    kmapper.visualize(
        grafo,
        path_html = html,
        title=f"Config B coloreado por {nombre}",
        X = X_numerico, X_names = nombres_caracteristicas,
        color_values = vec.reshape(-1, 1),
        color_function_name = ["mean"]
    )
    print("HTML:", html)

    valores_nodo = media_por_nodo(grafo, vec)

    dibujar_png(
        grafo, posiciones,
        valores_nodo = valores_nodo,
        titulo = f"Configuración B : {nombre}",
        salida = f"mapper_configB_{nombre}.png"
    )

print("Fin")

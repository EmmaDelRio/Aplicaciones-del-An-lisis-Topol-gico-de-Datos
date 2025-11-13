"""
MAPPER CONFIGURACIÓN B
"""

from pathlib import Path
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
from scipy.stats import zscore

# Entradas
RUTA_X = Path("X_mapper.npy")
assert RUTA_X.exists(), "No se encuentra X_mapper.npy"

# nombres 15 características
nombres_caract = [
    "seg1_media","seg1_desv","seg1_asimetria",
    "seg2_media","seg2_desv","seg2_asimetria",
    "seg3_media","seg3_desv","seg3_asimetria",
    "LF","HF","RR_media","RR_desv","amplitud_max","amplitud_min"
]

# carga
X = np.load(RUTA_X)
if X.ndim != 2:
    raise ValueError("X debe ser 2D")
assert X.shape[1] == 15, f"Esperaba 15 columnas, obtuve {X.shape[1]}"

# previo
escalador   = StandardScaler()
X_escalado  = escalador.fit_transform(X)

pca   = PCA(n_components = 10, random_state = 42)
X_pca = pca.fit_transform(X_escalado)
print("Varianza explicada acumulada (10 comp):", float(pca.explained_variance_ratio_.cumsum()[-1]))

# UMAP (posiciones 2D) 
umap_pos2d = umap.UMAP(
    n_neighbors  = 30,
    min_dist     = 0.05,
    n_components = 2,
    metric       = "euclidean",
    random_state = 42
).fit_transform(X_pca)

# función filtro: excentricidad kNN
def exc_knn(X_emb, k = 50):
    vecinos       = NearestNeighbors(n_neighbors = k + 1, metric = "euclidean", n_jobs = 1).fit(X_emb)
    distancias, _ = vecinos.kneighbors(X_emb, return_distance = True)
    distancias    = distancias[:, 1:]
    return distancias.mean(axis = 1)

k_exc   = 30
exc_1d  = exc_knn(X_pca, k = k_exc)
exc_z   = (exc_1d - np.mean(exc_1d)) / (np.std(exc_1d) + 1e-8)

# vectores características
ix_LF         = nombres_caract.index("LF")
ix_HF         = nombres_caract.index("HF")
ix_RRmedia    = nombres_caract.index("RR_media")
ix_RRdesv     = nombres_caract.index("RR_desv")
ix_amp_max    = nombres_caract.index("amplitud_max")
ix_amp_min    = nombres_caract.index("amplitud_min")
ix_s1_asim    = nombres_caract.index("seg1_asimetria")
ix_s2_asim    = nombres_caract.index("seg2_asimetria")
ix_s3_asim    = nombres_caract.index("seg3_asimetria")

LF_vector             = X[:, ix_LF]
HF_vector             = X[:, ix_HF]
RR_media_vector       = X[:, ix_RRmedia]
RR_desv_vector        = X[:, ix_RRdesv]
amplitud_max_vector   = X[:, ix_amp_max]
amplitud_min_vector   = X[:, ix_amp_min]
seg1_asim_vector      = X[:, ix_s1_asim]
seg2_asim_vector      = X[:, ix_s2_asim]
seg3_asim_vector      = X[:, ix_s3_asim]

# derivados
HF_seguro        = np.clip(HF_vector, 1e-8, np.percentile(HF_vector, 99.9))
LF_HF_ratio      = LF_vector / HF_seguro
rango_amp_vector = amplitud_max_vector - amplitud_min_vector

# para el grafo
def grafo_a_networkx(dic):
    G = nx.Graph()
    
    for id_nodo, indices in dic["nodes"].items():
        G.add_node(id_nodo, tamaño=len(indices))
    enlaces = dic.get("links", [])   
    for origen, destino in enlaces:
        G.add_edge(origen, destino)
    
    return G

def posiciones_umap(dic, umap_coords):
    pos = {}
    for id_nodo, idxs in dic["nodes"].items():
        pts = umap_coords[np.array(idxs)]
        pos[id_nodo] = pts.mean(axis = 0)
    return pos

def dibujar_png(dic, pos, valores_nodo = None, titulo = "", salida = "salida.png",
                vmin = None, vmax = None):

    G = grafo_a_networkx(dic)
    tamanos = [max(10, np.log10(G.nodes[n]["tamano"] + 1) * 60) for n in G.nodes()]

    fig, ax = plt.subplots(figsize = (10, 10))
    nx.draw_networkx_edges(G, pos, ax = ax, width = 1.2, edge_color = "#666666")

    if valores_nodo is None:
        nx.draw_networkx_nodes(G, pos, ax = ax, node_size = tamanos, node_color = "#3182bd")

    else:
        vals   = np.array([valores_nodo.get(n, np.nan) for n in G.nodes()])
        fin    = vals[np.isfinite(vals)]
        fill   = np.nanmedian(fin) if fin.size > 0 else 0.0
        vals   = np.nan_to_num(vals, nan = fill, posinf = fin.max() if fin.size > 0 else 0,
                                         neginf = fin.min() if fin.size > 0 else 0)

        nodos  = nx.draw_networkx_nodes(G, pos, ax = ax, node_size = tamanos,
                                        node_color = vals, cmap = "viridis",
                                        vmin = vmin, vmax = vmax)

        norm = mpl.colors.Normalize(
            vmin = np.min(vals) if vmin is None else vmin,
            vmax = np.max(vals) if vmax is None else vmax
        )

        sm = mpl.cm.ScalarMappable(cmap = nodos.get_cmap(), norm = norm)
        sm.set_array([])
        fig.colorbar(sm, ax = ax, shrink = 0.8).ax.set_ylabel("valor medio por nodo")

    ax.set_title(titulo)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(salida, dpi = 300)
    plt.close(fig)
    print("PNG:", salida)

def media_por_nodo(dic, vec):
    return {id_nodo: float(np.mean(vec[np.array(idxs)]))
            for id_nodo, idxs in dic["nodes"].items()}

# construcción Mapper
kmapper    = km.KeplerMapper(verbose = 1)
cubrimiento = km.Cover(n_cubes = 22, perc_overlap = 0.50)

clusterer = hdbscan.HDBSCAN(
    min_cluster_size = 8,
    min_samples      = 4,
    prediction_data  = True
)

grafo = kmapper.map(
    exc_z.reshape(-1, 1),
    X_pca,
    cover     = cubrimiento,
    clusterer = clusterer
)

# estadísticas grafo
G = grafo_a_networkx(grafo)
n_nodos   = G.number_of_nodes()
n_aristas = G.number_of_edges()
deg       = dict(G.degree())
aislados  = sum(1 for d in deg.values() if d == 0)

porc_aisl = 100.0 * aislados / n_nodos if n_nodos else 0.0
componentes = list(nx.connected_components(G))
aristas_por_nodo = n_aristas / n_nodos if n_nodos else float("nan")
mayor_comp       = max((len(c) for c in componentes), default = 0)
porc_mayor_comp  = mayor_comp / n_nodos * 100 if n_nodos else 0

print(
    f"nodos: {n_nodos}  aristas: {n_aristas}  comp: {len(componentes)}  "
    f"aislados: {aislados} ({porc_aisl:.1f}%)  edges/node: {aristas_por_nodo:.2f}  "
    f"mayor comp: {mayor_comp} ({porc_mayor_comp:.1f}%)"
)

# html
X_num  = pd.DataFrame(X, columns = nombres_caract).values
html_B = "mapper_configuracionB.html"

kmapper.visualize(
    grafo,
    path_html           = html_B,
    title               = "Configuración B",
    X                   = X_num,
    X_names             = nombres_caract,
    color_values        = exc_z.reshape(-1, 1),
    color_function_name = ["mean"]
)

print("HTML base:", html_B)

# png base
pos = posiciones_umap(grafo, umap_pos2d)
dibujar_png(grafo, pos, titulo = "Configuración B", salida = "mapper_configB_base.png")

# medidas extra
k_knn   = 30
vecinos = NearestNeighbors(n_neighbors = k_knn + 1, metric = "euclidean").fit(X_pca)
dist_knn_bruto, _ = vecinos.kneighbors(X_pca)
dist_knn = dist_knn_bruto[:, 1:].mean(axis = 1)

lof = LocalOutlierFactor(n_neighbors = k_knn, contamination = "auto")
lof.fit(X_pca)
score_lof = -lof.negative_outlier_factor_

extra = {
    "Excentricidad_Z" : exc_z,
    "Dist_kNN"        : dist_knn,
    "Score_LOF"       : score_lof
}

# repintado por características fisiológicas
fisio = {
    "RR_media" : RR_media_vector,
    "RR_desv"  : RR_desv_vector,
    "LF"       : LF_vector,
    "HF"       : HF_vector,
    "LF_HF"    : LF_HF_ratio
}

for nombre, vec in fisio.items():

    html = f"mapper_configB_{nombre}.html"
    kmapper.visualize(
        grafo,
        path_html           = html,
        title               = f"Config B  {nombre}",
        X                   = X_num,
        X_names             = nombres_caract,
        color_values        = vec.reshape(-1, 1),
        color_function_name = ["mean"]
    )
    print("HTML repintado:", html)

    nod = media_por_nodo(grafo, vec)
    dibujar_png(
        grafo, pos, valores_nodo = nod,
        titulo = f"Configuración B  {nombre}",
        salida = f"mapper_configB_{nombre}.png"
    )

# repintado morfología
morf = {
    "AMP_max"    : amplitud_max_vector,
    "AMP_min"    : amplitud_min_vector,
    "AMP_rango"  : rango_amp_vector,
    "SEG1_asim"  : seg1_asim_vector,
    "SEG2_asim"  : seg2_asim_vector,
    "SEG3_asim"  : seg3_asim_vector
}

for nombre, vec in morf.items():

    html = f"mapper_configB_{nombre}.html"
    kmapper.visualize(
        grafo,
        path_html           = html,
        title               = f"Config B {nombre}",
        X                   = X_num,
        X_names             = nombres_caract,
        color_values        = vec.reshape(-1, 1),
        color_function_name = ["mean"]
    )
    print("HTML repintado:", html)

    nod = media_por_nodo(grafo, vec)
    dibujar_png(
        grafo, pos, valores_nodo = nod,
        titulo = f"Configuración B  {nombre}",
        salida = f"mapper_configB_{nombre}.png"
    )

# repintado rareza
for nombre, vec in extra.items():

    html = f"mapper_configB_{nombre}.html"
    kmapper.visualize(
        grafo,
        path_html           = html,
        title               = f"Config B  {nombre}",
        X                   = X_num,
        X_names             = nombres_caract,
        color_values        = vec.reshape(-1, 1),
        color_function_name = ["mean"]
    )
    print("HTML coloreado:", html)

    nod = media_por_nodo(grafo, vec)
    dibujar_png(
        grafo, pos, valores_nodo = nod,
        titulo = f"Configuración B  {nombre}",
        salida = f"mapper_configB_{nombre}.png"
    )

# rareza compuesta
m_rareza        = np.vstack([exc_z, dist_knn, score_lof]).T
rareza_compuesta  = zscore(m_rareza, axis = 0, nan_policy = "omit").mean(axis = 1)

rareza_por_nodo   = media_por_nodo(grafo, rareza_compuesta)
tamano_por_nodo   = media_por_nodo(grafo, RR_desv_vector)
umbral_amp           = np.percentile(rango_amp_vector, 95)
alta_amp_por_nodo = media_por_nodo(grafo, (rango_amp_vector > umbral_amp).astype(float)) # para cada nodo: proporción de ptos que superan el umbral

def tamano_nodo(id_nodo):
    return len(grafo["nodes"][id_nodo])

id_nodo_raro = max(rareza_por_nodo, key = rareza_por_nodo.get)

nodos_naranja = [n for n, v in alta_amp_por_nodo.items() if v > 0.5]
id_nodo_naranja   = max(nodos_naranja, key = lambda n: rareza_por_nodo.get(n, -np.inf)) if nodos_naranja else None

if id_nodo_naranja is not None:
    elegido = id_nodo_raro if tamano_nodo(id_nodo_raro) <= tamano_nodo(id_nodo_naranja) else id_nodo_naranja
else:
    elegido = id_nodo_raro

print(f" nodo seleccionado: {elegido}")

# visualización rareza compuesta
G = grafo_a_networkx(grafo)

fig, ax = plt.subplots(figsize = (10, 10))
nx.draw_networkx_edges(G, pos, ax = ax, width = 1.2, edge_color = "#666666")

n_nodos_raw = np.array([tamano_por_nodo[n] for n in G.nodes()])
n_nodos_norm = 600 * (n_nodos_raw - n_nodos_raw.min()) / (n_nodos_raw.max() - n_nodos_raw.min() + 1e-8) + 50

colors = np.array([rareza_por_nodo[n] for n in G.nodes()])
pc = nx.draw_networkx_nodes(G, pos, ax = ax, node_size = n_nodos_norm, node_color = colors, cmap = "viridis")

cbar = fig.colorbar(pc, ax = ax, shrink = 0.8)
cbar.ax.set_ylabel("rareza compuesta (media nodo)")

nodos_extr = [n for n in G.nodes() if alta_amp_por_nodo[n] > 0.5] # mas del 50% de sus ptos tienen amplitud extrema
nx.draw_networkx_nodes(G, pos, nodelist = nodos_extr,
                      node_size = n_nodos_norm[[list(G.nodes()).index(n) for n in nodos_extr]],
                      node_color = "none", edgecolors = "orange", linewidths = 1.8, ax = ax)

ax.set_axis_off()
fig.tight_layout()
fig.savefig("mapper_configB_integrado.png", dpi = 300)
plt.close(fig)
print("PNG: mapper_configB_integrado.png")

# repintado por ratio LF/HF
lfhf_por_nodo      = media_por_nodo(grafo, LF_HF_ratio)
alta_amp_por_nodo2 = alta_amp_por_nodo
tamano_por_nodo2   = tamano_por_nodo

n_nodos_raw2 = np.array([tamano_por_nodo2[n] for n in G.nodes()])
n_nodos_norm2 = 600 * (n_nodos_raw2 - n_nodos_raw2.min()) / (n_nodos_raw2.max() - n_nodos_raw2.min() + 1e-8) + 100

colors2 = np.array([lfhf_por_nodo[n] for n in G.nodes()])

fig, ax = plt.subplots(figsize = (10, 10))
nx.draw_networkx_edges(G, pos, ax = ax, width = 1.2, edge_color = "#666666")

pc = nx.draw_networkx_nodes(G, pos, ax = ax, node_size = n_nodos_norm2, node_color = colors2, cmap = "viridis")
cbar = fig.colorbar(pc, ax = ax, shrink = 0.8)
cbar.ax.set_ylabel("LF/HF (media por nodo)")

nodos_extr = [n for n in G.nodes() if alta_amp_por_nodo2[n] > 0.5]
nx.draw_networkx_nodes(G, pos, nodelist = nodos_extr,
                      node_size = n_nodos_norm2[[list(G.nodes()).index(n) for n in nodos_extr]],
                      node_color = "none", edgecolors = "orange", linewidths = 2.5, ax = ax)

ax.set_title("Config B  color = LF/HF")
ax.set_axis_off()
fig.tight_layout()
fig.savefig("mapper_configB_integrado_LFHF.png", dpi = 300)
plt.close(fig)

print("Fin")



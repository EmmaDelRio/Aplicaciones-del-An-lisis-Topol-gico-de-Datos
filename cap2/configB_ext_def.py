"""
MAPPER CONFIGURACIÓN B - extendido
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

# nombres 15 caracteristicas (en castellano, sin tildes)
nombres_caract = [
    "seg1_media","seg1_desv","seg1_asim",
    "seg2_media","seg2_desv","seg2_asim",
    "seg3_media","seg3_desv","seg3_asim",
    "LF","HF","RR_media","RR_desv","amp_max","amp_min"
]

# carga
X = np.load(RUTA_X)
if X.ndim != 2:
    raise ValueError("X debe ser 2D")
assert X.shape[1] == 15, f"tendría que haber 15 columnas, hay {X.shape[1]}"

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

# funcion filtro: excentricidad kNN
def exc_knn(X_emb, k = 50):
    vecinos       = NearestNeighbors(n_neighbors = k + 1, metric = "euclidean", n_jobs = 1).fit(X_emb)
    distancias, _ = vecinos.kneighbors(X_emb, return_distance = True)
    distancias    = distancias[:, 1:]
    return distancias.mean(axis = 1)

k_exc   = 30
exc_1d  = exc_knn(X_pca, k = k_exc)
exc_z   = (exc_1d - np.mean(exc_1d)) / (np.std(exc_1d) + 1e-8)

# vectores caracteristicas (fisiologicos y morfologicos)
ix_LF       = nombres_caract.index("LF")
ix_HF       = nombres_caract.index("HF")
ix_RRmedia  = nombres_caract.index("RR_media")
ix_RRdesv   = nombres_caract.index("RR_desv")
ix_amp_max  = nombres_caract.index("amp_max")
ix_amp_min  = nombres_caract.index("amp_min")
ix_s1_asim  = nombres_caract.index("seg1_asim")
ix_s2_asim  = nombres_caract.index("seg2_asim")
ix_s3_asim  = nombres_caract.index("seg3_asim")

vector_LF          = X[:, ix_LF]
vector_HF          = X[:, ix_HF]
vector_RR_media    = X[:, ix_RRmedia]
vector_RR_desv     = X[:, ix_RRdesv]
vector_amp_max     = X[:, ix_amp_max]
vector_amp_min     = X[:, ix_amp_min]
vector_seg1_asim   = X[:, ix_s1_asim]
vector_seg2_asim   = X[:, ix_s2_asim]
vector_seg3_asim   = X[:, ix_s3_asim]

# derivados
HF_seguro    = np.clip(vector_HF, 1e-8, np.percentile(vector_HF, 99.9))
LF_HF_ratio  = vector_LF / HF_seguro
rango_amp    = vector_amp_max - vector_amp_min

# para el grafo
def grafo_a_networkx(dic):
    G = nx.Graph()
    for nid, miembros in dic["nodes"].items():
        G.add_node(nid, tamano = len(miembros))

    enlaces = dic.get("links", [])
    if isinstance(enlaces, dict):
        for a, vec in enlaces.items():
            for b in vec:
                G.add_edge(a, b)
    else:
        for link in enlaces:
            if isinstance(link, (list, tuple)) and len(link) >= 2:
                G.add_edge(link[0], link[1])
            elif isinstance(link, dict):
                a = link.get("source", link.get("from"))
                b = link.get("target", link.get("to"))
                if a is not None and b is not None:
                    G.add_edge(a, b)
    return G

def posiciones_umap(dic, umap_coords):
    pos = {}
    for nid, idxs in dic["nodes"].items():
        pts = umap_coords[np.array(idxs)]
        pos[nid] = pts.mean(axis = 0)
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
        vals  = np.array([valores_nodo.get(n, np.nan) for n in G.nodes()])
        fin   = vals[np.isfinite(vals)]
        fill  = np.nanmedian(fin) if fin.size > 0 else 0.0
        vals  = np.nan_to_num(
            vals,
            nan    = fill,
            posinf = fin.max() if fin.size > 0 else 0,
            neginf = fin.min() if fin.size > 0 else 0
        )

        nodos = nx.draw_networkx_nodes(
            G, pos, ax = ax,
            node_size  = tamanos,
            node_color = vals,
            cmap       = "viridis",
            vmin       = vmin,
            vmax       = vmax
        )

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
    return {
        nid: float(np.mean(vec[np.array(idxs)]))
        for nid, idxs in dic["nodes"].items()
    }

# construccion Mapper
mapeador    = km.KeplerMapper(verbose = 1)
cubrimiento = km.Cover(n_cubes = 22, perc_overlap = 0.50)

clusterer = hdbscan.HDBSCAN(
    min_cluster_size = 8,
    min_samples      = 4,
    prediction_data  = True
)

grafo = mapeador.map(
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

porc_aisl        = 100.0 * aislados / n_nodos if n_nodos else 0.0
componentes      = list(nx.connected_components(G))
aristas_por_nodo = n_aristas / n_nodos if n_nodos else float("nan")
mayor_comp       = max((len(c) for c in componentes), default = 0)
porc_mayor_comp  = mayor_comp / n_nodos * 100 if n_nodos else 0.0

print(
    f"nodos: {n_nodos}  aristas: {n_aristas}  comp: {len(componentes)}  "
    f"aislados: {aislados} ({porc_aisl:.1f}%)  edges/node: {aristas_por_nodo:.2f}  "
    f"mayor comp: {mayor_comp} ({porc_mayor_comp:.1f}%)"
)

# html
X_num  = pd.DataFrame(X, columns = nombres_caract).values
html_B = "mapper_configuracionB.html"

mapeador.visualize(
    grafo,
    path_html           = html_B,
    title               = "Configuracion B",
    X                   = X_num,
    X_names             = nombres_caract,
    color_values        = exc_z.reshape(-1, 1),
    color_function_name = ["mean"]
)

print("HTML base:", html_B)

# png base
pos = posiciones_umap(grafo, umap_pos2d)
dibujar_png(grafo, pos, titulo = "Configuracion B", salida = "mapper_configB_base.png")

# medidas extra de rareza
k_knn   = 30
vecinos = NearestNeighbors(n_neighbors = k_knn + 1, metric = "euclidean").fit(X_pca)
dist_knn_bruto, _ = vecinos.kneighbors(X_pca)
dist_knn = dist_knn_bruto[:, 1:].mean(axis = 1)

lof = LocalOutlierFactor(n_neighbors = k_knn, contamination = "auto")
lof.fit(X_pca)
score_lof = -lof.negative_outlier_factor_

extra = {
    "Excentricidad_Z" : exc_z,
    "kNN_Dist"        : dist_knn,
    "LOF_score"       : score_lof
}

# repintado por caracteristicas fisiologicas
fisio = {
    "RR_mean" : vector_RR_media,
    "RR_std"  : vector_RR_desv,
    "LF"      : vector_LF,
    "HF"      : vector_HF,
    "LF_HF"   : LF_HF_ratio
}

for nombre, vec in fisio.items():

    html = f"mapper_configB_{nombre}.html"
    mapeador.visualize(
        grafo,
        path_html           = html,
        title               = f"Config B – repintado por {nombre}",
        X                   = X_num,
        X_names             = nombres_caract,
        color_values        = vec.reshape(-1, 1),
        color_function_name = ["mean"]
    )
    print("HTML repintado:", html)

    valores_nodo = media_por_nodo(grafo, vec)
    dibujar_png(
        grafo, pos, valores_nodo = valores_nodo,
        titulo = f"Configuracion B – {nombre}",
        salida = f"mapper_configB_{nombre}.png"
    )

# repintado morfologia
morf = {
    "AMP_max"    : vector_amp_max,
    "AMP_min"    : vector_amp_min,
    "AMP_range"  : rango_amp,
    "SEG1_skew"  : vector_seg1_asim,
    "SEG2_skew"  : vector_seg2_asim,
    "SEG3_skew"  : vector_seg3_asim
}

for nombre, vec in morf.items():

    html = f"mapper_configB_{nombre}.html"
    mapeador.visualize(
        grafo,
        path_html           = html,
        title               = f"Config B – repintado morfologia {nombre}",
        X                   = X_num,
        X_names             = nombres_caract,
        color_values        = vec.reshape(-1, 1),
        color_function_name = ["mean"]
    )
    print("HTML repintado:", html)

    valores_nodo = media_por_nodo(grafo, vec)
    dibujar_png(
        grafo, pos, valores_nodo = valores_nodo,
        titulo = f"Configuracion B – {nombre}",
        salida = f"mapper_configB_{nombre}.png"
    )

# repintado rareza (Excentricidad_Z, kNN_Dist, LOF_score)
for nombre, vec in extra.items():

    html = f"mapper_configB_{nombre}.html"
    mapeador.visualize(
        grafo,
        path_html           = html,
        title               = f"Config B – rareza {nombre}",
        X                   = X_num,
        X_names             = nombres_caract,
        color_values        = vec.reshape(-1, 1),
        color_function_name = ["mean"]
    )
    print("HTML coloreado:", html)

    valores_nodo = media_por_nodo(grafo, vec)
    dibujar_png(
        grafo, pos, valores_nodo = valores_nodo,
        titulo = f"Configuracion B – {nombre}",
        salida = f"mapper_configB_{nombre}.png"
    )

# rareza compuesta, tamano por nodo y rango de amplitud
matriz_rareza      = np.vstack([exc_z, dist_knn, score_lof]).T
puntuacion_rareza  = zscore(matriz_rareza, axis = 0, nan_policy = "omit").mean(axis = 1)

rareza_por_nodo    = media_por_nodo(grafo, puntuacion_rareza)
tamano_por_nodo    = media_por_nodo(grafo, vector_RR_desv)
umbral_amp         = np.percentile(rango_amp, 95)
nodo_alto_rango    = media_por_nodo(grafo, (rango_amp > umbral_amp).astype(float))

def tamano_nodo(nid):
    return len(grafo["nodes"][nid])

nid_raro = max(rareza_por_nodo, key = rareza_por_nodo.get)

nodos_naranja = [n for n, v in nodo_alto_rango.items() if v > 0.5]
nid_orange    = max(nodos_naranja, key = lambda n: rareza_por_nodo.get(n, -np.inf)) if nodos_naranja else None

if nid_orange is not None:
    elegido = nid_raro if tamano_nodo(nid_raro) <= tamano_nodo(nid_orange) else nid_orange
else:
    elegido = nid_raro

print(f"[CANDIDATO] nodo seleccionado: {elegido}")

# visualizacion de rareza compuesta
G = grafo_a_networkx(grafo)

fig, ax = plt.subplots(figsize = (10, 10))
nx.draw_networkx_edges(G, pos, ax = ax, width = 1.2, edge_color = "#666666")

tamanos_brutos = np.array([tamano_por_nodo[n] for n in G.nodes()])
tamanos2 = 600 * (tamanos_brutos - tamanos_brutos.min()) / (tamanos_brutos.max() - tamanos_brutos.min() + 1e-8) + 50

colores = np.array([rareza_por_nodo[n] for n in G.nodes()])
pc = nx.draw_networkx_nodes(
    G, pos, ax = ax,
    node_size  = tamanos2,
    node_color = colores,
    cmap       = "viridis"
)

cbar = fig.colorbar(pc, ax = ax, shrink = 0.8)
cbar.ax.set_ylabel("rareza compuesta (media nodo)")

edge_nodes = [n for n in G.nodes() if nodo_alto_rango[n] > 0.5]
nx.draw_networkx_nodes(
    G, pos, nodelist = edge_nodes,
    node_size  = tamanos2[[list(G.nodes()).index(n) for n in edge_nodes]],
    node_color = "none",
    edgecolors = "orange",
    linewidths = 1.8,
    ax         = ax
)

ax.set_axis_off()
fig.tight_layout()
fig.savefig("mapper_configB_integrado.png", dpi = 300)
plt.close(fig)
print("PNG: mapper_configB_integrado.png")

# repintado por ratio LF/HF
lfhf_por_nodo   = media_por_nodo(grafo, LF_HF_ratio)
nodo_alto_rango2 = media_por_nodo(grafo, (rango_amp > umbral_amp).astype(float))
tamano_por_nodo2 = tamano_por_nodo

tamanos_brutos2 = np.array([tamano_por_nodo2[n] for n in G.nodes()])
tamanos3 = 600 * (tamanos_brutos2 - tamanos_brutos2.min()) / (tamanos_brutos2.max() - tamanos_brutos2.min() + 1e-8) + 100

colores2 = np.array([lfhf_por_nodo[n] for n in G.nodes()])

fig, ax = plt.subplots(figsize = (10, 10))
nx.draw_networkx_edges(G, pos, ax = ax, width = 1.2, edge_color = "#666666")

pc2 = nx.draw_networkx_nodes(
    G, pos, ax = ax,
    node_size  = tamanos3,
    node_color = colores2,
    cmap       = "viridis"
)
cbar2 = fig.colorbar(pc2, ax = ax, shrink = 0.8)
cbar2.ax.set_ylabel("LF/HF (media por nodo)")

edge_nodes = [n for n in G.nodes() if nodo_alto_rango2[n] > 0.5]
nx.draw_networkx_nodes(
    G, pos, nodelist = edge_nodes,
    node_size  = tamanos3[[list(G.nodes()).index(n) for n in edge_nodes]],
    node_color = "none",
    edgecolors = "orange",
    linewidths = 2.5,
    ax         = ax
)

ax.set_title("Config B – color = LF/HF")
ax.set_axis_off()
fig.tight_layout()
fig.savefig("mapper_configB_integrado_LFHF.png", dpi = 300)
plt.close(fig)

print("Fin")

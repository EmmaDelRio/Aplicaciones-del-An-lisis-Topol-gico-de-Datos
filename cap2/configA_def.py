"""
MAPPER CONFIGURACIÓN A
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import hdbscan
import kmapper as km
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

# Entradas
RUTA_X = Path("X_mapper.npy")
assert RUTA_X.exists(), "Archivo no encontrado"

X = np.load(RUTA_X)
assert X.ndim == 2 and X.shape[1] == 15, f"Se esperaba [n,15], se obtuvo {X.shape}"

# Nombres de las 15 características
nombres_caracteristicas = [
    "seg1_media", "seg1_desv", "seg1_asimetria",
    "seg2_media", "seg2_desv", "seg2_asimetria", 
    "seg3_media", "seg3_desv", "seg3_asimetria",
    "LF", "HF", "RR_media", "RR_desv", "amp_max", "amp_min"
]

# Escalado y PCA
escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)
pca = PCA(n_components=10, random_state=42)
X_pca = pca.fit_transform(X_escalado)

print("Varianza explicada acumulada:", pca.explained_variance_ratio_.cumsum()[-1])

# Función filtro: UMAP 2D
lente_umap = umap.UMAP(
    n_neighbors=80,
    min_dist=0.10,
    n_components=2, 
    metric="euclidean",
    random_state=42
).fit_transform(X_pca)

# Cover 14×14, 50% + HDBSCAN
mapeador = km.KeplerMapper(verbose=1)
cover = km.Cover(n_cubes=14, perc_overlap=0.50)
clusterer = hdbscan.HDBSCAN(min_cluster_size=40, min_samples=5, prediction_data=True)

grafo = mapeador.map(
    lente_umap,
    X_pca,
    cover=cover,
    clusterer=clusterer
)

# Datos numéricos para las estadísticas
X_numerico = pd.DataFrame(X, columns=nombres_caracteristicas).values
nombres_X = nombres_caracteristicas[:]

# HTML base sin tooltips
html_base = "mapper_configA.html"
mapeador.visualize(
    grafo,
    path_html=html_base,
    title="Configuración A",
    X=X_numerico,
    X_names=nombres_X
)
print("HTML completado")

def calcular_posiciones_nodos(diccionario_grafo, lente_2d):
    posiciones = {}
    for id_nodo, indices in diccionario_grafo["nodos"].items():
        puntos = lente_2d[np.array(indices)]
        posiciones[id_nodo] = puntos.mean(axis=0)
    return posiciones

def convertir_grafo_a_nx(diccionario_grafo):
    G = nx.Graph()
    for id_nodo, miembros in diccionario_grafo["nodos"].items():
        G.add_node(id_nodo, tamano=len(miembros))
    enlaces = diccionario_grafo.get("enlaces", [])
    if isinstance(enlaces, dict):
        for a, vecinos in enlaces.items():
            for b in vecinos:
                G.add_edge(a, b)
    else:
        for enlace in enlaces:
            if isinstance(enlace, (list, tuple)) and len(enlace) >= 2:
                G.add_edge(enlace[0], enlace[1])
            elif isinstance(enlace, dict):
                a = enlace.get("source", enlace.get("from"))
                b = enlace.get("target", enlace.get("to"))
                if a is not None and b is not None:
                    G.add_edge(a, b)
    return G

def dibujar_png(diccionario_grafo, posiciones, valores_nodos=None,
                titulo="", salida="salida.png", vmin=None, vmax=None,
                tamano_titulo=16):
    G = convertir_grafo_a_nx(diccionario_grafo)
    tamanos = [max(10, np.log10(G.nodes[n]["tamano"] + 1) * 60) for n in G.nodes()]
    fig, ax = plt.subplots(figsize=(10, 10))

    nx.draw_networkx_edges(G, posiciones, ax=ax, width=1.2, edge_color="#666666")

    if valores_nodos is None:
        nx.draw_networkx_nodes(G, posiciones, ax=ax, node_size=tamanos, node_color="#3182bd")
    else:
        valores = np.array([valores_nodos.get(n, np.nan) for n in G.nodes()])
        finitos = valores[np.isfinite(valores)]
        if finitos.size == 0:
            valores[:] = 0.0
        else:
            relleno = np.nanmedian(finitos)
            valores = np.nan_to_num(valores, nan=relleno, posinf=finitos.max(), neginf=finitos.min())

        mapeable = nx.draw_networkx_nodes(
            G, posiciones, ax=ax,
            node_size=tamanos, node_color=valores,
            cmap="viridis", vmin=vmin, vmax=vmax
        )
        norm = mpl.colors.Normalize(
            vmin=mapeable.get_array().min() if vmin is None else vmin,
            vmax=mapeable.get_array().max() if vmax is None else vmax
        )
        sm = mpl.cm.ScalarMappable(cmap=mapeable.get_cmap(), norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.8).ax.set_ylabel("valor medio por nodo")

    ax.set_title(titulo, fontsize=tamano_titulo)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(salida, dpi=300)
    plt.close(fig)
    print(f"PNG: {salida}")

def agregar_media_por_nodo(diccionario_grafo, vector):
    return {id_nodo: float(np.mean(vector[np.array(indices)]))
            for id_nodo, indices in diccionario_grafo["nodos"].items()}

# Resumen del grafo
G = convertir_grafo_a_nx(grafo)
num_nodos = G.number_of_nodes()
num_aristas = G.number_of_edges()
grados = dict(G.degree())
aislados = sum(1 for d in grados.values() if d == 0)
porc_aislados = 100.0 * aislados / num_nodos if num_nodos else 0.0
componentes = list(nx.connected_components(G))
aristas_por_nodo = num_aristas / num_nodos if num_nodos else float('nan')
mayor_componente = max((len(c) for c in componentes), default=0)

print(
    f"nodos: {num_nodos}  aristas: {num_aristas}  componentes: {len(componentes)}  "
    f"aislados: {aislados} ({porc_aislados:.1f}%)  aristas/nodo: {aristas_por_nodo:.2f}  "
    f"componente mayor: {mayor_componente} nodos ({(mayor_componente/num_nodos*100 if num_nodos else 0):.1f}%)"
)

# Posiciones a partir de UMAP
posiciones = calcular_posiciones_nodos(grafo, lente_umap)

# Grafo base
dibujar_png(
    grafo, posiciones,
    titulo="Configuración A",
    tamano_titulo=25,
    salida="A_base.png"
)

# Colores por características
campos_color = ["RR_media", "RR_desv", "LF", "HF", "LF_HF_ratio", "amp_max", "amp_min"]

if "LF" in nombres_caracteristicas and "HF" in nombres_caracteristicas:
    lf = X[:, nombres_caracteristicas.index("LF")]
    hf = X[:, nombres_caracteristicas.index("HF")]
    hf_seguro = np.clip(hf, 1e-8, np.percentile(hf, 99.9))
    relacion_lf_hf = lf / hf_seguro
else:
    relacion_lf_hf = None

for nombre in campos_color:
    if nombre == "LF_HF_ratio":
        if relacion_lf_hf is None:
            continue
        vector = relacion_lf_hf
    else:
        vector = X[:, nombres_caracteristicas.index(nombre)]

    html = f"mapper_configA_{nombre}.html"
    mapeador.visualize(
        grafo,
        path_html=html,
        title=f"Configuración A coloreado por {nombre}",
        X=X_numerico,
        X_names=nombres_X,
        color_values=vector.reshape(-1, 1),
        color_function_name=["mean"]
    )
    print(f"HTML coloreado: {html}")

    valores_nodos = agregar_media_por_nodo(grafo, vector)
    dibujar_png(
        grafo, posiciones,
        valores_nodos=valores_nodos,
        titulo=f"Configuración A : {nombre}",
        salida=f"mapper_configA_{nombre}.png"
    )

print("Fin")

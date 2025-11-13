"""
APLICACIONES DEL TDA - DETECCIÓN ÓRBITAS CAÓTICAS
SISTEMA DE LORENZ - embedding de Takens (solo visualización)
"""
import numpy as np                          
import matplotlib.pyplot as plt  
import pandas as pd 
from ripser import ripser
from persim import plot_diagrams
from sklearn.preprocessing import MinMaxScaler

# CONFIGURACIÓN GLOBAL

# PI y preprocesado
RESOLUCION               = 32     # 32x32 = 1024 características
CANALES_POR_DIMENSION    = True   # True -> por separado H0 y H1, False -> mezcla todo
NORMALIZAR_PI            = False  # True -> normaliza cada imagen por su suma, False -> valores absolutos


# CARGAR Y PREPARAR CONJUNTO DE DATOS

def cargar_lorenz(ruta, umbral_le = 0.01):
    # Carga el conjunto de datos del sistema de Lorenz y asigna etiquetas según el exponente de Lyapunov
    
    dataset = []

    with open(ruta, "r") as f:
        for linea in f:
            valores = list(map(float, linea.strip().split()))
            
            if len(valores) != 7 + 3000:
                print("Línea con longitud inesperada, se omite.")
                continue

            # Extrae condiciones iniciales, parámetros y Lyapunov
            x0, y0, z0 = valores[0:3]
            sigma, rho, beta = valores[3:6]
            lyap = valores[6]

            # Extrae la serie temporal (x, y, z)
            serie_raw = valores[7:]
            serie = [serie_raw[i:i+3] for i in range(0, len(serie_raw), 3)]

            # Etiqueta según el exponente de Lyapunov
            etiqueta = 1 if lyap > umbral_le else 0

            muestra = {
                "x0": x0,
                "y0": y0,
                "z0": z0,
                "sigma": sigma,
                "rho": rho,
                "beta": beta,
                "lyapunov": lyap,
                "label": etiqueta,
                "serie": serie
            }

            dataset.append(muestra)

    return dataset

def embed_takens_manual(serie, dimension = 3, retardo = 1):
    # Aplica la incrustación de Takens a una sola serie (x, y, z)
    incrustada = []
    for i in range(len(serie) - (dimension - 1) * retardo):
        punto = []
        for j in range(dimension):
            punto.extend(serie[i + j * retardo])  # concatena [x, y, z]
        incrustada.append(punto)
    return np.array(incrustada)

def aplicar_takens(dataset, dimension = 3, retardo = 1):
    # Aplica la incrustación de Takens a cada muestra del conjunto
    for item in dataset:
        serie = item['serie']
        emb = embed_takens_manual(serie, dimension = dimension, retardo = retardo)
        item['embedded'] = emb
    return dataset

# APLICAR TDA

def preparar_para_tda(dataset):
    # Normaliza las nubes de puntos de cada muestra para TDA
    for item in dataset:
        scaler = MinMaxScaler()
        puntos_normalizados = scaler.fit_transform(item['embedded'])
        item['embedded'] = puntos_normalizados  
    return dataset

def calcular_persistencia(dataset, tamano_lote = 300, dimensiones_homologia = (0, 1)):
    # Calcula los diagramas de persistencia para cada serie temporal
    
    resultados = []
    total_lotes = (len(dataset) - 1) // tamano_lote + 1
    
    for num_lote in range(total_lotes):
        inicio = num_lote * tamano_lote
        fin    = inicio + tamano_lote
        lote   = dataset[inicio:fin]
        
        print(f" Lote {num_lote + 1}/{total_lotes} ({len(lote)} muestras)")
        
        for item in lote:
            X = item['embedded']
            if len(X) > 1000:
                X = X[::2]
            dgms = ripser(X, maxdim = max(dimensiones_homologia))['dgms']
            item['diagram'] = dgms
        
        resultados.extend(lote)
        print(f" Lote {num_lote + 1} completado")
    
    return resultados

def diagramas_a_dict(diagrama):
    # Convierte una lista de diagramas en un diccionario {dim: diagrama}
    if isinstance(diagrama, list):
        return {i: diagrama[i] for i in range(len(diagrama))}
    return dict(diagrama)

def nacimiento_persistencia(puntos):
    # Calcula los pares (nacimiento, persistencia)
    if puntos is None or len(puntos) == 0:
        return None
    pts = puntos[np.isfinite(puntos[:, 1])]
    if len(pts) == 0:
        return None
    nac = pts[:, 0]
    pers = np.maximum(pts[:, 1] - pts[:, 0], 0.0)
    return np.column_stack([nac, pers])

def rangos_globales(dataset, dims = (0, 1), margen = 0.05):
    # Calcula los rangos globales de nacimiento y persistencia
    todos = []
    for item in dataset:
        dct = diagramas_a_dict(item['diagram'])
        for dim in dims:
            bp = nacimiento_persistencia(dct.get(dim))
            if bp is not None and len(bp) > 0:
                todos.append(bp)
    if not todos:
        return (0.0, 1.0), (0.0, 1.0)
    todos = np.vstack(todos)
    bmin, pmin = todos.min(axis = 0)
    bmax, pmax = todos.max(axis = 0)
    mb = margen * max(bmax - bmin, 1e-9)
    mp = margen * max(pmax - pmin, 1e-9)
    return (bmin - mb, bmax + mb), (pmin - mp, pmax + mp)

def vectorizar_pi(
    dataset, dims = (0, 1),
    resolucion = RESOLUCION, 
    canales_por_dimension = CANALES_POR_DIMENSION,
    sigma_px = 1.0,
    normalizar = NORMALIZAR_PI, 
    dtype = np.float32,
    rango_nacimiento = None, 
    rango_persistencia = None
):
    # Genera imágenes de persistencia (PIs) sumando gaussianas en coordenadas de píxel
   
    if rango_nacimiento is None or rango_persistencia is None:
        b_range, p_range = rangos_globales(dataset, dims = dims, margen = 0.05)
    else:
        b_range, p_range = rango_nacimiento, rango_persistencia

    bmin, bmax = b_range
    pmin, pmax = p_range
    if bmax <= bmin: bmax = bmin + 1e-6
    if pmax <= pmin: pmax = pmin + 1e-6

    ys, xs = np.mgrid[0:resolucion, 0:resolucion]
    dos_sigma2 = 2.0 * (sigma_px ** 2)

    def agregar_gaussianas(puntos_bp):
        # Suma gaussianas sobre una malla de píxeles
        
        if puntos_bp is None or len(puntos_bp) == 0:
            return np.zeros((resolucion, resolucion), dtype = dtype)
        bx = (puntos_bp[:, 0] - bmin) / (bmax - bmin) * (resolucion - 1)
        py = (puntos_bp[:, 1] - pmin) / (pmax - pmin) * (resolucion - 1)
        imagen = np.zeros((resolucion, resolucion), dtype = np.float64)
        for x0, y0 in zip(bx, py):
            d2 = (xs - x0) ** 2 + (ys - y0) ** 2
            imagen += np.exp(-d2 / max(dos_sigma2, 1e-12))
        return imagen.astype(dtype)

    for item in dataset:
        dct = diagramas_a_dict(item['diagram'])
        if canales_por_dimension:
            canales = []
            for dim in dims:
                bp = nacimiento_persistencia(dct.get(dim))
                img = agregar_gaussianas(bp)
                if normalizar and img.sum() > 0:
                    img = (img / img.sum()).astype(dtype)
                canales.append(img)
            persimg = np.stack(canales, axis = -1)
        else:
            todos_bp = []
            for dim in dims:
                bp = nacimiento_persistencia(dct.get(dim))
                if bp is not None and len(bp) > 0:
                    todos_bp.append(bp)
            if todos_bp:
                todos_bp = np.vstack(todos_bp)
            else:
                todos_bp = np.empty((0, 2))
            img = agregar_gaussianas(todos_bp)
            if normalizar and img.sum() > 0:
                img = (img / img.sum()).astype(dtype)
            persimg = img[..., None]

        item['persimg'] = persimg
        item['vector']  = persimg.ravel().astype(dtype)

    return dataset

def ilustracion():
    # Genera ejemplos visuales de series caóticas y no caóticas con Takens y diagramas de persistencia
    
    print("Inicio")
    print("Cargando conjunto de datos")

    COMBINACIONES = [(10, 2), (6, 2)]
    FINAL_DIM, FINAL_DELAY = COMBINACIONES[0]
    PUNTOS_A_TOMAR = 500  # Últimos 500 puntos

    dataset = cargar_lorenz("C:/Users/HOME/Desktop/TFG/lorenz/LS_TRAIN_Data_Paper_norm.txt",
                            umbral_le = 0.01)

    # Extrae los últimos 500 puntos de una serie regular y una caótica
    for item in dataset:
        if item['label'] == 0 and 'serie_nocaos' not in locals():
            serie_nocaos = item['serie'][-PUNTOS_A_TOMAR:]
        elif item['label'] == 1 and 'serie_caos' not in locals():
            serie_caos = item['serie'][-PUNTOS_A_TOMAR:]
        if 'serie_nocaos' in locals() and 'serie_caos' in locals():
            break
        
    # Figura 1: series temporales
    print("Figura 1")
    plt.figure(figsize = (10, 4))
    plt.subplot(1, 2, 1)
    plt.plot([x[0] for x in serie_nocaos], lw = 1)
    plt.xlabel("Time")
    plt.ylabel("x")
    plt.title("No caótica", fontsize = 18)
    plt.subplot(1, 2, 2)
    plt.plot([x[0] for x in serie_caos], lw = 1, color = 'red')
    plt.xlabel("Time")
    plt.ylabel("x")
    plt.title("Caótica", fontsize = 18)
    plt.tight_layout()
    plt.show()

    # Figura 2: diagramas de persistencia para varias combinaciones Takens
    print("Figura 2")
    fig, axs = plt.subplots(2, 2, figsize = (10, 8))
    
    for fila, (dim, ret) in enumerate(COMBINACIONES):
       
        # No caótica 
        print("No caótica")
        emb_noc = embed_takens_manual(serie_nocaos, dimension = dim, retardo = ret)  
        emb_noc = MinMaxScaler().fit_transform(emb_noc)
        dgms_noc = ripser(emb_noc, maxdim = 2)['dgms']
        plot_diagrams(dgms_noc, show = False, ax = axs[fila, 0])
        axs[fila, 0].set_title(f"No caótica d={dim}, τ={ret}")

        # Caótica 
        print("Caótica")
        emb_c = embed_takens_manual(serie_caos, dimension = dim, retardo = ret)      
        emb_c = MinMaxScaler().fit_transform(emb_c)
        dgms_c = ripser(emb_c, maxdim = 2)['dgms']
        plot_diagrams(dgms_c, show = False, ax = axs[fila, 1])
        axs[fila, 1].set_title(f"Caótica  d={dim}, τ={ret}")

    plt.tight_layout()
    plt.show()
    
    print("Fin")


# EJECUCIÓN DEL PROGRAMA PRINCIPAL

if __name__ == "__main__":
    ilustracion()


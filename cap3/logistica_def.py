"""
APLICACIONES DEL TDA - DETECCIÓN ÓRBITAS CAÓTICAS (castellano)

ECUACIÓN LOGÍSTICA, CNN desarrollada para comparar con el artículo
"""

import numpy as np
from ripser import ripser
from persim import plot_diagrams
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import collections
import matplotlib.pyplot as plt
import hashlib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# CONFIGURACIÓN GLOBAL

# Reproducibilidad
def fijar_semillas(semilla: int = 42):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
fijar_semillas(42)

# Splits por condiciones iniciales
x0_entrenamiento = [0.3, 0.9]
x0_validacion    = [0.55]
x0_prueba        = [0.8]

# Para generar los r_vals
def crear_malla_r(n: int = 6000, r_min: float = 0.0, r_max: float = 4.0, endpoint: bool = False):
    return np.linspace(r_min, r_max, n, endpoint=endpoint)

malla_r = crear_malla_r(n=10000)

# Longitud, transitorio y umbral para LE
longitud_serie = 1000
transitorio = 1000
umbral_le = 0.1

# PI y preprocesado
RESOLUCION = 32  # 32x32 = 1024 características
CANALES_POR_DIMENSION = True  # True -> por separado H0 y H1, False -> mezcla todo
NORMALIZAR_PI = False  # True -> normaliza cada imagen por su suma, False -> valores absolutos

# GENERAR CONJUNTO DE DATOS

def ec_logistica(x0, r, longitud = longitud_serie, trans = transitorio):
    n = longitud + trans
    x = np.zeros(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return x[trans:]

def exp_lyapunov(serie, r, indice_inicio: int = 0, eps: float = 1e-12):
    # si la serie entrante tiene los primeros trans -> indice_inicio=trans
    
    n = len(serie)
    if n < indice_inicio + 500:
        return 0.0
    derivada = r * (1 - 2 * serie[indice_inicio:])
    abs_der = np.abs(derivada)
    vals = np.where(abs_der > eps, abs_der, eps)  # evitar log(0)
    return np.mean(np.log(vals))

def generar_conjunto_datos(x0_vals, r_vals,
                           longitud = longitud_serie, 
                           trans = transitorio,
                           umbral=umbral_le):
    
    dataset = []
    for r in r_vals:
        for x0 in x0_vals:
            serie = ec_logistica(x0, r, longitud = longitud, trans = trans)
            if len(serie) < longitud:
                continue
            le = exp_lyapunov(serie, r, indice_inicio = 0)
            etiqueta = 1 if le > umbral else 0
            dataset.append({
                'serie': serie,
                'label': etiqueta,
                'r': r,
                'x0': x0
            })
    return dataset

def balancear_clases(dataset, n_por_clase = None, semilla = 42):
    
    rng = np.random.default_rng(semilla)
    cls0 = [d for d in dataset if d['label'] == 0]
    cls1 = [d for d in dataset if d['label'] == 1]
    if len(cls0) == 0 or len(cls1) == 0:
        return dataset  # no se puede balancear

    if n_por_clase is None:
        k = min(len(cls0), len(cls1))
    else:
        k = min(n_por_clase, len(cls0), len(cls1))

    sel0 = rng.choice(cls0, size=k, replace=False).tolist()
    sel1 = rng.choice(cls1, size=k, replace=False).tolist()
    combinado = sel0 + sel1
    rng.shuffle(combinado)
    return combinado

def hash_serie(serie, decimales = 4):
    q = np.round(serie, decimals=decimales)
    return hashlib.md5(q.tobytes()).hexdigest()

def eliminar_duplicados(dataset, decimales = 4):
    # Para que no haya series parecidas en el mismo conjunto
    
    vistos = set()
    filtrado = []
    for item in dataset:
        h = hash_serie(item['serie'], decimales)
        if h not in vistos:
            vistos.add(h)
            filtrado.append(item)
    print(f"De {len(dataset)} a {len(filtrado)}")
    return filtrado

def eliminar_duplicados_entre_grupos(dataset_ref, dataset_objetivo, decimales = 4):
    # Para que no haya series muy parecidas en los diferentes conjuntos
    
    hashes_ref = set(hash_serie(item['serie'], decimales) for item in dataset_ref)
    filtrado = []
    eliminadas = 0
    for item in dataset_objetivo:
        h = hash_serie(item['serie'], decimales)
        if h not in hashes_ref:
            filtrado.append(item)
        else:
            eliminadas += 1
    print(f"Eliminadas {eliminadas} muestras")
    return filtrado

def cargar_conjunto(ruta_archivo, longitud = 1000, umbral = 0.1, decimales = 4):
    # carga el conjunto de datos de test del artículo
    # formato: Initial condition, parameter a, first LE, time series
    
    dataset = []

    try:
        with open(ruta_archivo, 'r') as f:
            lineas = f.readlines()

        for i, linea in enumerate(lineas):
            partes = linea.strip().split()
            try:
                x0 = float(partes[0].strip())
                r = float(partes[1].strip())
                exp_lyap = float(partes[2].strip())

                valores_serie = partes[3:]
                serie = np.array([float(val.strip()) for val in valores_serie if val.strip()])

                # Crea etiqueta
                etiqueta = 1 if exp_lyap > umbral else 0

                dataset.append({
                    'serie': serie,
                    'label': etiqueta,
                    'r': r,
                    'x0': x0,
                    'exp_lyapunov': exp_lyap  # guarda el valor real del LE
                })

            except ValueError as e:
                print(f"Error en línea {i+1}: {e}")
                continue

    except Exception as e:
        print(f"Error leyendo archivo: {e}")
        return []

    if len(dataset) > 0:
        distribucion_clases(dataset, nombre="Dataset Externo")
    else:
        print("vacío")

    return dataset

def embed_takens_manual(serie, dimension=3, retardo=1):
    incrustada = []
    for i in range(len(serie) - (dimension - 1) * retardo):
        punto = [serie[i + j * retardo] for j in range(dimension)]
        incrustada.append(punto)
    return np.array(incrustada)

def aplicar_takens(dataset, dim_incrustacion=3, retardo=1):
    for item in dataset:
        serie = item['serie']
        emb = embed_takens_manual(serie, dimension=dim_incrustacion, retardo=retardo)
        item['embedded'] = emb
    return dataset

# APLICAR TDA

def calcular_persistencia(dataset, tamano_lote = 300, dimensiones_homologia = (0, 1)):
    
    resultados = []
    total_lotes = (len(dataset) - 1) // tamano_lote + 1

    for num_lote in range(total_lotes):
        inicio = num_lote * tamano_lote
        fin = inicio + tamano_lote
        lote = dataset[inicio:fin]

        print(f" Lote {num_lote + 1}/{total_lotes} ({len(lote)} muestras)") # indicador

        for item in lote:
            X = item['embedded']
            # Submuestreo opcional:
            # if len(X) > 1000:
            #     X = X[::2]
            X = MinMaxScaler().fit_transform(X)
            dgms = ripser(X, maxdim=max(dimensiones_homologia))['dgms']
            item['diagram'] = dgms

        resultados.extend(lote)
        print(f" Lote {num_lote + 1} completado") # indicador

    return resultados

def diagramas_a_dict(diagrama):
    # lista/diccionario -> diccionario (ripser da listas)
    
    if isinstance(diagrama, list):
        return {i: diagrama[i] for i in range(len(diagrama))}
    return dict(diagrama)

def nacimiento_persistencia(puntos):
    # Devuelve (birth, persistence>=0)
    
    if puntos is None or len(puntos) == 0:
        return None
    # Filtra muertes infinitas y NaN
    pts = puntos[np.isfinite(puntos[:, 1])]
    if len(pts) == 0:
        return None
    nac = pts[:, 0]
    pers = np.maximum(pts[:, 1] - pts[:, 0], 0.0)
    return np.column_stack([nac, pers])

def rangos_globales(dataset, dims = (0, 1), margen = 0.05):
    
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
    bmin, pmin = todos.min(axis=0)
    bmax, pmax = todos.max(axis=0)
    # márgenes para no recortar puntos del borde
    mb = margen * max(bmax - bmin, 1e-9)
    mp = margen * max(pmax - pmin, 1e-9)
    return (bmin - mb, bmax + mb), (pmin - mp, pmax + mp)

def vectorizar_pi(
    dataset, dims = (0, 1),
    resolucion = RESOLUCION,
    canales_por_dimension = CANALES_POR_DIMENSION,
    sigma_px = 1.0,
    normalizar = NORMALIZAR_PI,
    dtype=np.float32,
    rango_nacimiento = None,
    rango_persistencia = None
):
    # Genera PIs sumando gaussianas en coordenadas de píxel

    # Rangos birth/pers
    if rango_nacimiento is None or rango_persistencia is None:
        b_range, p_range = rangos_globales(dataset, dims = dims, margen = 0.05)
    else:
        b_range, p_range = rango_nacimiento, rango_persistencia

    bmin, bmax = b_range
    pmin, pmax = p_range
    if bmax <= bmin:
        bmax = bmin + 1e-6
    if pmax <= pmin:
        pmax = pmin + 1e-6

    # Malla de píxeles
    ys, xs = np.mgrid[0:resolucion, 0:resolucion]
    dos_sigma2 = 2.0 * (sigma_px ** 2)

    def agregar_gaussianas(puntos_bp):
        if puntos_bp is None or len(puntos_bp) == 0:
            return np.zeros((resolucion, resolucion), dtype=dtype)
        # (birth, pers) -> coords de píxel
        bx = (puntos_bp[:, 0] - bmin) / (bmax - bmin) * (resolucion - 1)
        py = (puntos_bp[:, 1] - pmin) / (pmax - pmin) * (resolucion - 1)
        imagen = np.zeros((resolucion, resolucion), dtype=np.float64)
        for x0, y0 in zip(bx, py):
            d2 = (xs - x0) ** 2 + (ys - y0) ** 2
            imagen += np.exp(-d2 / max(dos_sigma2, 1e-12))
        return imagen.astype(dtype)

    for item in dataset:
        dct = diagramas_a_dict(item['diagram'])
        if canales_por_dimension:
            canales = []
            for dim in dims:
                bp = nacimiento_persistencia(dct.get(dim))  # (nacimiento, persistencia>=0)
                img = agregar_gaussianas(bp)
                if normalizar and img.sum() > 0:
                    img = (img / img.sum()).astype(dtype)
                canales.append(img)
            persimg = np.stack(canales, axis=-1)
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
        item['vector'] = persimg.ravel().astype(dtype)

    return dataset

def distribucion_clases(dataset, nombre = "Train"):
    etiquetas = [item['label'] for item in dataset]
    print(f"\n DISTRIBUCIÓN DE CLASES DE {nombre}")
    print(collections.Counter(etiquetas))

# DEFINICIÓN Y ENTRENAMIENTO DE LA CNN

def preparar_entrenamiento(dataset):
    
    # Extrae imágenes/vectores y etiquetas -> formato necesario para entrenamiento
    X_lista = [item['persimg'] for item in dataset]  # si queremos 1D, usar 'vector'
    X = np.stack(X_lista, axis=0)
    y = np.array([item['label'] for item in dataset])

    # comprobación
    print(f"Forma de X: {X.shape}")
    print(f"Forma de y: {y.shape}")
    return X, y

class Tiny2DBackbone(nn.Module):
    
    def __init__(self, in_channels = 2):
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 5, kernel_size = 3, dilation = 2, padding = 2, bias = True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(5, 10, kernel_size = 3, dilation = 4, padding = 4, bias = True)
        self.act2 = nn.ReLU(inplace = True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # invarianza a HxW

    def forward(self, x):  # x: (N, C, H, W)
    
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.gap(x).view(x.size(0), -1)  # (N, 10)
        return x

class PersistenceCNN(nn.Module):
    
    def __init__(self, input_channels = 2, num_classes = 2):
        
        super().__init__()
        self.backbone = Tiny2DBackbone(in_channels = input_channels)
        self.fc = nn.Linear(10, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean = 0.0, std = 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feats = self.backbone(x)     # (N, 10)
        logits = self.fc(feats)      # (N, num_classes)
        return logits

def preparar_cargador_datos(X, y, tamano_lote = 32, mezclar = False):
    # (N, H, W, C) -> (N, C, H, W) para PyTorch
    
    X_t = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
    y_t = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size = tamano_lote, shuffle = mezclar)

def entrenar_cnn(modelo, cargador_entrenamiento, cargador_validacion, epocas = 2000, lr = 8e-3,
                 dispositivo = "cpu", paciencia = 100, weight_decay = 1e-5, pesos_clase = None,
                 verbose = True):

    modelo.to(dispositivo)
    optimizador = optim.Adam(modelo.parameters(), lr = lr, weight_decay = weight_decay)
    criterio = nn.CrossEntropyLoss(weight=(pesos_clase.to(dispositivo) if pesos_clase is not None else None))

    mejor_val_loss = float("inf")
    contador_paciencia = 0
    mejor_estado = None

    historial = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoca in range(epocas):
        modelo.train()
        perdida_acum, correctas, total = 0.0, 0, 0
        for X_batch, y_batch in cargador_entrenamiento:
            X_batch, y_batch = X_batch.to(dispositivo), y_batch.to(dispositivo)

            optimizador.zero_grad()
            salidas = modelo(X_batch)
            perdida = criterio(salidas, y_batch)
            perdida.backward()
            optimizador.step()

            perdida_acum += perdida.item() * X_batch.size(0)
            correctas += (salidas.argmax(1) == y_batch).sum().item()
            total += y_batch.size(0)

        train_loss = perdida_acum / max(1, total)
        train_acc = correctas / max(1, total)

        modelo.eval()
        val_correctas, val_total = 0, 0
        val_perdida_sum = 0.0
        with torch.no_grad():
            for X_batch, y_batch in cargador_validacion:
                X_batch, y_batch = X_batch.to(dispositivo), y_batch.to(dispositivo)
                salidas = modelo(X_batch)
                perdida = criterio(salidas, y_batch)
                val_perdida_sum += perdida.item() * X_batch.size(0)
                val_correctas += (salidas.argmax(1) == y_batch).sum().item()
                val_total += y_batch.size(0)

        val_loss = val_perdida_sum / max(1, val_total)
        val_acc = val_correctas / max(1, val_total)

        historial["train_loss"].append(train_loss)
        historial["val_loss"].append(val_loss)
        historial["train_acc"].append(train_acc)
        historial["val_acc"].append(val_acc)

        if verbose and (epoca % 10 == 0 or epoca == epocas - 1):
            print(f"Época {epoca+1:4d}/{epocas} - "
                  f"Train {train_loss:.4f}/{train_acc:.3f} - "
                  f"Val {val_loss:.4f}/{val_acc:.3f}")

        # early stopping (val_loss)
        if val_loss < mejor_val_loss - 1e-6:
            mejor_val_loss = val_loss
            contador_paciencia = 0
            mejor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
        else:
            contador_paciencia += 1

        if contador_paciencia >= paciencia:
            if verbose:
                print(f"Early stopping en época {epoca+1} | mejor val_loss: {mejor_val_loss:.4f}")
            break

    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
        if verbose:
            print("cargado mejor modelo")

    return modelo, historial

def graficar_perdida_precision(historial):
    # Gráfico de accuracy y pérdida de entrenamiento y validación
    
    epocas = range(1, len(historial["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))

    # pérdida
    plt.subplot(1, 2, 1)
    plt.plot(epocas, historial["train_loss"], label = "Train Loss")
    plt.plot(epocas, historial["val_loss"], label = "Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Val Loss")
    plt.legend()

    # accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epocas, historial["train_acc"], label = "Train Acc")
    plt.plot(epocas, historial["val_acc"], label = "Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train/Val Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

def evaluar_modelo(modelo, cargador, dispositivo = "cpu", criterio = None):
    
    modelo.eval()
    todas_etiquetas, todas_preds, todas_probs = [], [], []
    perdida_total, muestras_totales = 0.0, 0

    with torch.no_grad():
        for X_batch, y_batch in cargador:
            X_batch, y_batch = X_batch.to(dispositivo), y_batch.to(dispositivo)
            salidas = modelo(X_batch)

            if criterio is not None:
                perdida = criterio(salidas, y_batch)
                perdida_total += perdida.item() * X_batch.size(0)
                muestras_totales += X_batch.size(0)

            probs = torch.softmax(salidas, dim=1)
            preds = probs.argmax(dim=1)

            todas_etiquetas.extend(y_batch.cpu().numpy())
            todas_preds.extend(preds.cpu().numpy())
            todas_probs.extend(probs[:, 1].cpu().numpy())

    todas_etiquetas = np.array(todas_etiquetas)
    todas_preds = np.array(todas_preds)
    todas_probs = np.array(todas_probs)

    perdida_media = perdida_total / muestras_totales if (criterio is not None and muestras_totales > 0) else None

    # métricas
    cm = confusion_matrix(todas_etiquetas, todas_preds)
    acc_global = (todas_preds == todas_etiquetas).mean()
    acc_por_clase = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)

    # matriz de confusión
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["no caótico", "caótico"],
                yticklabels=["no caótico", "caótico"])
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta real")
    plt.show()

    print("\nClasificación:")
    print(classification_report(todas_etiquetas, todas_preds, target_names=["no caótico", "caótico"]))
    print(f"Exactitud global: {acc_global:.4f}")
    print(f"Exactitud por clase -> no caótico: {acc_por_clase[0]:.4f} | caótico: {acc_por_clase[1]:.4f}")
    if perdida_media is not None:
        print(f"Pérdida (Val/Test Loss): {perdida_media:.4f}")

    # histograma de probabilidades (clase 1)
    plt.figure(figsize=(6, 4))
    plt.hist(todas_probs[todas_etiquetas == 0], bins=20, alpha=0.7, label="No caótico")
    plt.hist(todas_probs[todas_etiquetas == 1], bins=20, alpha=0.7, label="Caótico")
    plt.xlabel("Probabilidad de ser caótico")
    plt.ylabel("Número de muestras")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=1)
    plt.tight_layout()
    plt.show()

    return todas_etiquetas, todas_preds, todas_probs

# PROGRAMA PRINCIPAL (1 semilla)

def principal():
    print("GENERACIÓN DE CONJUNTOS DE DATOS")
    fijar_semillas(42)

    malla_r = crear_malla_r(n = 10000)

    print("Conjunto de datos para entrenamiento")
    conjunto_entrenamiento = generar_conjunto_datos(x0_entrenamiento, malla_r)
    conjunto_entrenamiento = eliminar_duplicados(conjunto_entrenamiento, decimales = 4)
    conjunto_entrenamiento = balancear_clases(conjunto_entrenamiento)

    print("Conjunto de datos para validación")
    conjunto_validacion = generar_conjunto_datos(x0_validacion, malla_r)
    conjunto_validacion = eliminar_duplicados(conjunto_validacion, decimales = 4)
    conjunto_validacion = eliminar_duplicados_entre_grupos(conjunto_entrenamiento, conjunto_validacion, decimales = 4)
    conjunto_validacion = balancear_clases(conjunto_validacion)

    print("Conjunto de datos para test")
    conjunto_prueba = generar_conjunto_datos(x0_prueba, malla_r)
    conjunto_prueba = eliminar_duplicados(conjunto_prueba, decimales = 4)
    conjunto_prueba = eliminar_duplicados_entre_grupos(conjunto_entrenamiento, conjunto_prueba, decimales = 4)
    conjunto_prueba = eliminar_duplicados_entre_grupos(conjunto_validacion, conjunto_prueba, decimales = 4)
    conjunto_prueba = balancear_clases(conjunto_prueba)

    print(f"Final Train: {len(conjunto_entrenamiento)} muestras")
    print(f"Final Validation: {len(conjunto_validacion)} muestras")
    print(f"Final Test: {len(conjunto_prueba)} muestras")

    # TDA
    print("Aplicando embedding de Takens")
    conjunto_entrenamiento = aplicar_takens(conjunto_entrenamiento)
    conjunto_validacion = aplicar_takens(conjunto_validacion)
    conjunto_prueba = aplicar_takens(conjunto_prueba)

    print("Calculando persistencia")
    conjunto_entrenamiento = calcular_persistencia(conjunto_entrenamiento)
    conjunto_validacion = calcular_persistencia(conjunto_validacion)
    conjunto_prueba = calcular_persistencia(conjunto_prueba)

    print("Vectorizando")
    rango_nacimiento, rango_persistencia = rangos_globales(conjunto_entrenamiento, dims = (0, 1))

    conjunto_entrenamiento = vectorizar_pi(conjunto_entrenamiento, dims = (0, 1), resolucion = 32,
                                           canales_por_dimension = True, sigma_px = 1.0, normalizar = False,
                                           rango_nacimiento = rango_nacimiento, rango_persistencia = rango_persistencia)
    conjunto_validacion = vectorizar_pi(conjunto_validacion, dims = (0, 1), resolucion = 32,
                                        canales_por_dimension = True, sigma_px = 1.0, normalizar = False,
                                        rango_nacimiento = rango_nacimiento, rango_persistencia = rango_persistencia)
    conjunto_prueba = vectorizar_pi(conjunto_prueba, dims = (0, 1), resolucion = 32,
                                    canales_por_dimension = True, sigma_px = 1.0, normalizar = False,
                                    rango_nacimiento = rango_nacimiento, rango_persistencia = rango_persistencia)

    distribucion_clases(conjunto_entrenamiento, nombre = "Entrenamiento (σ=1.0)")
    distribucion_clases(conjunto_validacion, nombre = "Validación (σ=1.0)")
    distribucion_clases(conjunto_prueba, nombre = "Test (σ=1.0)")

    X_train, y_train = preparar_entrenamiento(conjunto_entrenamiento)
    X_val, y_val = preparar_entrenamiento(conjunto_validacion)
    X_test, y_test = preparar_entrenamiento(conjunto_prueba)

    cargador_entrenamiento = preparar_cargador_datos(X_train, y_train, tamano_lote = 128, mezclar = True)
    cargador_validacion = preparar_cargador_datos(X_val, y_val, tamano_lote = 128, mezclar = False)
    cargador_prueba = preparar_cargador_datos(X_test, y_test, tamano_lote = 128, mezclar = False)

    print("\n Entrenando modelo")
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    canales_entrada = X_train.shape[-1]
    modelo = PersistenceCNN(input_channels = canales_entrada, num_classes = 2)

    modelo, historial = entrenar_cnn(modelo, cargador_entrenamiento, cargador_validacion,
                                     epocas = 2000, lr = 8e-3, dispositivo = dispositivo,
                                     paciencia = 100, weight_decay = 1e-5,
                                     verbose = True)

    graficar_perdida_precision(historial)

    criterio = nn.CrossEntropyLoss()

    # DESCOMENTAR SI SE QUIERE HACER PARA 1 SEMILLA Y EL CONJUNTO DE DATOS GENERADO
    # COMENTAR SI SE QUIERE HACER PARA EL CONJUNTO DE TEST DEL ARTÍCULO
    print("\n Evaluando en test")
    todas_etiquetas, todas_preds, todas_probs = evaluar_modelo(modelo, cargador_prueba,
                                                               dispositivo = dispositivo,
                                                               criterio = criterio)

    torch.save(modelo.state_dict(), 'persistence_cnn_model.pth')
    print("Modelo guardado como 'persistence_cnn_model.pth'")
    print("\n fin ")

    print("\n")
    print("ANALIZANDO CONJUNTO DE TEST DEL ARTÍCULO")

    ruta_dataset = "C:/Users/Emma/Desktop/TFG_def/códigos_tfg/cap 3/LM_TEST_Data_Paper.txt"

    print(f"Cargando dataset: {ruta_dataset}")
    dataset_externo = cargar_conjunto(ruta_dataset)

    print("Embedding de Takens")
    dataset_externo = aplicar_takens(dataset_externo)

    print("Persistencia")
    dataset_externo = calcular_persistencia(dataset_externo)

    print("Vectorizando")
    dataset_externo = vectorizar_pi(dataset_externo, dims = (0, 1), resolucion = 32,
                                    canales_por_dimension = True, sigma_px = 1.0, normalizar = False,
                                    rango_nacimiento = rango_nacimiento, rango_persistencia = rango_persistencia)

    # COMENTAR SI NO SE QUIERE TRABAJAR CON EL CONJUNTO DE TEST DEL ARTÍCULO
    """
    X_externo, y_externo = preparar_entrenamiento(dataset_externo)
    cargador_externo = preparar_cargador_datos(X_externo, y_externo, tamano_lote = 128, mezclar = False)

    print("\nEvaluando")
    todas_etiquetas, todas_preds, todas_probs = evaluar_modelo(modelo, cargador_externo,
                                                               dispositivo = dispositivo, criterio = criterio)

    print("\n")
    print("ANÁLISIS FINAL DEL CONJUNTO DE TEST DEL ARTÍCULO")

    analisis_final(modelo, cargador_externo, dataset_externo, dispositivo = dispositivo)

    torch.save(modelo.state_dict(), 'persistence_cnn_model.pth')
    print("Modelo guardado como 'persistence_cnn_model.pth'")
    print("\n PROGRAMA COMPLETADO ")
    """

def analisis_final(modelo, cargador_prueba, dataset_prueba, dispositivo = "cpu"):
    # Análisis del conjunto de test del artículo + comparación
    
    print("\n1. EVALUACIÓN DEL MODELO TDA+CNN")
    criterio = nn.CrossEntropyLoss()
    todas_etiquetas, todas_preds, todas_probs = evaluar_modelo(modelo, cargador_prueba,
                                                               dispositivo = dispositivo, criterio = criterio)

    print("\n2. COMPARACIÓN DIRECTA CON ARTÍCULO")
    comparar_con_resultados_articulo(todas_etiquetas, todas_preds)

    print("\n3. PRECISIÓN POR RANGOS DE LYAPUNOV")
    graficar_precision_por_lyapunov(dataset_prueba, todas_preds, todas_etiquetas)

    print("\n4. ANÁLISIS DE ERRORES")
    analizar_casos_dificiles(dataset_prueba, todas_preds, todas_etiquetas)

    print("\n")
    print("fin")

def comparar_con_resultados_articulo(etiquetas_reales, predicciones):
    # Comparación directa con los resultados del artículo
    
    resultados_articulo = {
        'CNN': {'test_accuracy': 99.41, 'nonchaotic_acc': 99.71, 'chaotic_acc': 99.11},
    }

    nuestra_acc = np.mean(predicciones == etiquetas_reales) * 100

    mask_no_caotico = (etiquetas_reales == 0)
    mask_caotico = (etiquetas_reales == 1)

    acc_no_caotico = np.mean(predicciones[mask_no_caotico] == etiquetas_reales[mask_no_caotico]) * 100
    acc_caotico = np.mean(predicciones[mask_caotico] == etiquetas_reales[mask_caotico]) * 100

    # Gráfico de comparación
    fig, ax = plt.subplots(1, 2, figsize = (15, 6))

    metodos = ['TDA+CNN', 'CNN [25]']
    # global
    accuracies = [nuestra_acc, resultados_articulo['CNN']['test_accuracy']]

    colores = ['red', 'blue', 'green', 'orange']
    bars = ax[0].bar(metodos, accuracies, color = colores)
    ax[0].set_ylabel('Accuracy (%)')
    ax[0].set_title(' Accuracy Total', fontsize = 20)
    ax[0].tick_params(axis = 'x', rotation = 45)
    ax[0].set_ylim(90, 100)

    for bar, acc in zip(bars, accuracies):
        altura = bar.get_height()
        ax[0].text(bar.get_x() + bar.get_width() / 2., altura + 0.1,
                   f'{acc:.2f}%', ha='center', va='bottom')

    # por clase
    acc_clase = {
        'No caótico': [acc_no_caotico, resultados_articulo['CNN']['nonchaotic_acc']],
        'Caótico': [acc_caotico, resultados_articulo['CNN']['chaotic_acc']]
    }

    x = np.arange(len(metodos))
    width = 0.35

    bars1 = ax[1].bar(x - width / 2, acc_clase['No caótico'], width, label = 'No caótico', alpha = 0.8)
    bars2 = ax[1].bar(x + width / 2, acc_clase['Caótico'], width, label = 'Caótico', alpha = 0.8)

    ax[1].set_ylabel('Accuracy (%)')
    ax[1].set_title('Accuracy por Clase', fontsize = 20)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(metodos, rotation = 45)
    ax[1].legend(loc = 'lower center', bbox_to_anchor = (0.5, 0.01))
    ax[1].set_ylim(90, 100)

    for barras in [bars1, bars2]:
        for bar in barras:
            altura = bar.get_height()
            ax[1].text(bar.get_x() + bar.get_width() / 2., altura + 0.1,
                       f'{altura:.2f}%', ha = 'center', va = 'bottom', fontsize = 8)

    plt.tight_layout()
    plt.show()

    print(f"TDA+CNN: {nuestra_acc:.2f}%")
    print(f"CNN artículo: {resultados_articulo['CNN']['test_accuracy']:.2f}%")
    print(f"Diferencia: {nuestra_acc - resultados_articulo['CNN']['test_accuracy']:+.2f}%")

def graficar_precision_por_lyapunov(dataset, predicciones, etiquetas_reales):
    # Precisión por rangos de Lyapunov
    
    valores_lyapunov = np.array([item.get('exp_lyapunov', 0) for item in dataset])

    bins = np.linspace(valores_lyapunov.min(), valores_lyapunov.max(), 8)
    idx_bins = np.digitize(valores_lyapunov, bins)

    accuracies = []
    for idx in range(1, len(bins)):
        mask = (idx_bins == idx)
        if np.sum(mask) > 0:
            acc_bin = np.mean(predicciones[mask] == etiquetas_reales[mask]) * 100
            accuracies.append(acc_bin)
        else:
            accuracies.append(0)

    plt.figure(figsize=(10, 6))
    barras = plt.bar(range(len(accuracies)), accuracies,
                     color=['red' if acc < 85 else 'orange' if acc < 95 else 'green' for acc in accuracies])

    plt.xlabel('Rango de Exponente de Lyapunov')
    plt.ylabel('Precisión (%)')
    plt.axhline(y=90, color='r', linestyle='--', alpha=0.7, label='Umbral 90%')
    plt.legend()
    plt.grid(True, alpha=0.3)

    etiquetas_eje = [f'{bins[i]:.2f}\na\n{bins[i+1]:.2f}' for i in range(len(accuracies))]
    plt.xticks(range(len(accuracies)), etiquetas_eje)

    for bar, acc in zip(barras, accuracies):
        if acc > 0:  # mostrar valores para bins no vacíos
            plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                     f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

def analizar_casos_dificiles(dataset, predicciones, etiquetas_reales, n_casos = 3):
    # casos con errores
    
    mascara_errores = (predicciones != etiquetas_reales)
    indices_error = np.where(mascara_errores)[0]

    if len(indices_error) == 0:
        print("0 errores")
        return

    # casos representativos
    muestras_error = indices_error[:min(n_casos, len(indices_error))]

    fig, axes = plt.subplots(1, len(muestras_error), figsize = (5 * len(muestras_error), 4))

    # por si solo hay 1 error
    if len(muestras_error) == 1:
        axes = [axes]

    for i, idx in enumerate(muestras_error):
        item = dataset[idx]
        
        axes[i].plot(item['serie'][:150], 'b-', alpha = 0.7, linewidth = 2)
        axes[i].set_title(
            f'r={item["r"]:.3f}, LE={item.get("exp_lyapunov", 0):.3f}'
        )
        axes[i].set_xlabel('Paso temporal')
        axes[i].set_ylabel('Valor')
        axes[i].grid(True, alpha = 0.3)

    plt.tight_layout()
    plt.show()

# PARA 5 SEMILLAS (robustez y estabilidad)

def experimento_diferentes_semillas(n_ejecuciones = 5, semillas = None,
                                    tamano_lote = 128,
                                    resolucion = 32,
                                    sigma_px = 1.0,
                                    epocas = 2000, lr = 8e-3,
                                    paciencia = 100, decimales = 4):

    if semillas is None:
        semillas = [42, 678, 3002, 43654, 81828][:n_ejecuciones]

    # Acumuladores
    accs_train, accs_val, accs_test = [], [], []
    losses_train, losses_val, losses_test = [], [], []
    accs_test_clase0, accs_test_clase1 = [], []

    for ejec in range(1, n_ejecuciones + 1):
        sem = semillas[ejec - 1]
        print(f"\nEXPERIMENTO {ejec}/{n_ejecuciones} - Semilla {sem}")
        fijar_semillas(sem)

        # generación datos
        malla_r = crear_malla_r(n = 10000)

        conjunto_entrenamiento = generar_conjunto_datos(x0_entrenamiento, malla_r)
        conjunto_entrenamiento = eliminar_duplicados(conjunto_entrenamiento, decimales = decimales)
        conjunto_entrenamiento = balancear_clases(conjunto_entrenamiento)

        conjunto_validacion = generar_conjunto_datos(x0_validacion, malla_r)
        conjunto_validacion = eliminar_duplicados(conjunto_validacion, decimales = decimales)
        conjunto_validacion = eliminar_duplicados_entre_grupos(conjunto_entrenamiento, conjunto_validacion, decimales = decimales)
        conjunto_validacion = balancear_clases(conjunto_validacion)

        conjunto_prueba = generar_conjunto_datos(x0_prueba, malla_r)
        conjunto_prueba = eliminar_duplicados(conjunto_prueba, decimales = decimales)
        conjunto_prueba = eliminar_duplicados_entre_grupos(conjunto_entrenamiento, conjunto_prueba, decimales = decimales)
        conjunto_prueba = eliminar_duplicados_entre_grupos(conjunto_validacion, conjunto_prueba, decimales = decimales)
        conjunto_prueba = balancear_clases(conjunto_prueba)

        print(f"Final Entrenamiento: {len(conjunto_entrenamiento)} | Val: {len(conjunto_validacion)} | Test: {len(conjunto_prueba)}")

        # TDA
        conjunto_entrenamiento = aplicar_takens(conjunto_entrenamiento)
        conjunto_validacion = aplicar_takens(conjunto_validacion)
        conjunto_prueba = aplicar_takens(conjunto_prueba)

        conjunto_entrenamiento = calcular_persistencia(conjunto_entrenamiento)
        conjunto_validacion = calcular_persistencia(conjunto_validacion)
        conjunto_prueba = calcular_persistencia(conjunto_prueba)

        # rangos solo de train
        rango_nacimiento, rango_persistencia = rangos_globales(conjunto_entrenamiento, dims = (0, 1))

        # vectorizar
        conjunto_entrenamiento = vectorizar_pi(conjunto_entrenamiento, dims = (0, 1),
                                               resolucion = resolucion,
                                               canales_por_dimension = True,
                                               sigma_px = sigma_px, normalizar = False,
                                               rango_nacimiento = rango_nacimiento, rango_persistencia = rango_persistencia)
        conjunto_validacion = vectorizar_pi(conjunto_validacion, dims = (0, 1),
                                            resolucion = resolucion,
                                            canales_por_dimension = True,
                                            sigma_px = sigma_px, normalizar = False,
                                            rango_nacimiento = rango_nacimiento, rango_persistencia = rango_persistencia)
        conjunto_prueba = vectorizar_pi(conjunto_prueba, dims = (0, 1),
                                        resolucion = resolucion,
                                        canales_por_dimension = True,
                                        sigma_px = sigma_px, normalizar = False,
                                        rango_nacimiento = rango_nacimiento, rango_persistencia = rango_persistencia)

        X_train, y_train = preparar_entrenamiento(conjunto_entrenamiento)
        X_val, y_val = preparar_entrenamiento(conjunto_validacion)
        X_test, y_test = preparar_entrenamiento(conjunto_prueba)

        cargador_entrenamiento = preparar_cargador_datos(X_train, y_train, tamano_lote=tamano_lote, mezclar = True)
        cargador_validacion = preparar_cargador_datos(X_val, y_val, tamano_lote=tamano_lote, mezclar = False)
        cargador_prueba = preparar_cargador_datos(X_test, y_test, tamano_lote=tamano_lote, mezclar = False)

        # cnn y entrenamiento
        dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        canales_entrada = X_train.shape[-1]
        modelo = PersistenceCNN(input_channels = canales_entrada, num_classes = 2)

        modelo, historial = entrenar_cnn(modelo, cargador_entrenamiento, cargador_validacion,
                                         epocas=epocas, lr = lr, dispositivo = dispositivo,
                                         paciencia = paciencia, verbose = True)

        accs_train.append(historial["train_acc"][-1])
        accs_val.append(historial["val_acc"][-1])
        losses_train.append(historial["train_loss"][-1])
        losses_val.append(historial["val_loss"][-1])

        # evaluación test
        criterio = nn.CrossEntropyLoss()
        modelo.eval()
        perdida_total, total_n = 0.0, 0
        todas_preds, todas_etiquetas = [], []
        with torch.no_grad():
            for xb, yb in cargador_prueba:
                xb, yb = xb.to(dispositivo), yb.to(dispositivo)
                logits = modelo(xb)
                perdida = criterio(logits, yb)
                perdida_total += perdida.item() * yb.size(0)
                total_n += yb.size(0)
                todas_preds.extend(logits.argmax(1).cpu().numpy())
                todas_etiquetas.extend(yb.cpu().numpy())
        test_loss = perdida_total / max(1, total_n)
        losses_test.append(test_loss)

        todas_preds = np.array(todas_preds)
        todas_etiquetas = np.array(todas_etiquetas)
        test_acc_global = (todas_preds == todas_etiquetas).mean()
        cm = confusion_matrix(todas_etiquetas, todas_preds)
        acc_por_clase = np.diag(cm) / np.maximum(cm.sum(axis = 1), 1)

        accs_test.append(test_acc_global)
        accs_test_clase0.append(float(acc_por_clase[0]))
        accs_test_clase1.append(float(acc_por_clase[1]))

        print(f"Run {ejec} | Test Acc: {test_acc_global:.4f} | "
              f"Clase0: {acc_por_clase[0]:.4f} | Clase1: {acc_por_clase[1]:.4f} | "
              f"Test Loss: {test_loss:.4f}")

    # resumen
    def media_std(arr):
        return float(np.mean(arr)), float(np.std(arr))

    print("\n")
    print("RESULTADOS FINALES")
    print(f"Train Acc : {media_std(accs_train)[0]:.4f} ± {media_std(accs_train)[1]:.4f}")
    print(f"Val   Acc : {media_std(accs_val)[0]:.4f}   ± {media_std(accs_val)[1]:.4f}")
    print(f"Test  Acc : {media_std(accs_test)[0]:.4f}  ± {media_std(accs_test)[1]:.4f}")
    print(f"Train Loss: {media_std(losses_train)[0]:.4f} ± {media_std(losses_train)[1]:.4f}")
    print(f"Val   Loss: {media_std(losses_val)[0]:.4f}   ± {media_std(losses_val)[1]:.4f}")
    print(f"Test  Loss: {media_std(losses_test)[0]:.4f}  ± {media_std(losses_test)[1]:.4f}")
    print("\nRESULTADOS POR CLASES (TEST)")
    print(f"Clase 0 Acc: {media_std(accs_test_clase0)[0]:.4f} ± {media_std(accs_test_clase0)[1]:.4f}")
    print(f"Clase 1 Acc: {media_std(accs_test_clase1)[0]:.4f} ± {media_std(accs_test_clase1)[1]:.4f}")

# PARA LA PRESENTACIÓN DEL CAPÍTULO

def ilustracion():
    
    # parámetros para la figura
    longitud = 500
    trans = 200
    r_caos, r_nocaos = 4.0, 3.5
    x0 = 0.2
    combinaciones = [(6, 2), (3, 1)]

    # series y Lyapunov
    serie_caos = ec_logistica(x0, r_caos, longitud = longitud, trans = trans)
    serie_nocaos = ec_logistica(x0, r_nocaos, longitud = longitud, trans = trans)
    print(f"Lyapunov (r={r_nocaos}): {exp_lyapunov(serie_nocaos, r_nocaos):.3f}")
    print(f"Lyapunov (r={r_caos}):   {exp_lyapunov(serie_caos, r_caos):.3f}")

    # figura 1: series
    fragmento = 100
    plt.figure(figsize = (10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(serie_nocaos[:fragmento], lw = 1)
    plt.title(f"Serie no caótica (r={r_nocaos})")
    plt.xlabel("Iteraciones")
    plt.ylabel("x")
    plt.subplot(1, 2, 2)
    plt.plot(serie_caos[:fragmento], lw = 1)
    plt.title(f"Serie caótica (r={r_caos})")
    plt.xlabel("Iteraciones")
    plt.ylabel("x")
    plt.tight_layout()
    plt.show()

    # figura 2: diagramas de persistencia
    fig, axs = plt.subplots(2, 2, figsize = (10, 8))
    for fila, (dim, ret) in enumerate(combinaciones):
        X_noc = embed_takens_manual(serie_nocaos, dimension = dim, retardo = ret)
        X_noc = MinMaxScaler().fit_transform(X_noc)
        dgms_noc = ripser(X_noc, maxdim = 1)['dgms']

        X_c = embed_takens_manual(serie_caos, dimension = dim, retardo = ret)
        X_c = MinMaxScaler().fit_transform(X_c)
        dgms_c = ripser(X_c, maxdim=1)['dgms']

        plot_diagrams(dgms_noc, show = False, ax = axs[fila, 0])
        axs[fila, 0].set_title(f"No caótica  d={dim}, $\\tau={ret}$")
        plot_diagrams(dgms_c, show = False, ax = axs[fila, 1])
        axs[fila, 1].set_title(f"Caótica     d={dim}, $\\tau={ret}$")
    plt.tight_layout()
    plt.show()

    # PIs
    dim_pi, retardo_pi = combinaciones[0]  # 1ª combinación de parámetros

    X_nocaos = embed_takens_manual(serie_nocaos, dimension = dim_pi, retardo = retardo_pi)
    X_nocaos = MinMaxScaler().fit_transform(X_nocaos)
    dgms_nocaos = ripser(X_nocaos, maxdim = 1)['dgms']

    X_caos = embed_takens_manual(serie_caos, dimension = dim_pi, retardo = retardo_pi)
    X_caos = MinMaxScaler().fit_transform(X_caos)
    dgms_caos = ripser(X_caos, maxdim = 1)['dgms']

    dataset = [
        {"diagram": dgms_nocaos, "label": 0},  # no caótica
        {"diagram": dgms_caos,   "label": 1}   # caótica
    ]

    dataset = vectorizar_pi(
        dataset,
        dims = (0, 1),
        resolucion = 32,
        canales_por_dimension = True,
        sigma_px = 1.2,     # más bajo = más nítido; más alto = más suave
        normalizar = False
    )

    # imágenes por canal
    img_nocaos_H0 = np.array(dataset[0]['persimg'][:, :, 0], dtype = float)
    img_nocaos_H1 = np.array(dataset[0]['persimg'][:, :, 1], dtype = float)
    img_caos_H0 = np.array(dataset[1]['persimg'][:, :, 0], dtype = float)
    img_caos_H1 = np.array(dataset[1]['persimg'][:, :, 1], dtype = float)

    # pintado
    def vmax_compartido(imgs, q = 98):
        vec = np.concatenate([im.ravel() for im in imgs])
        vec = vec[np.isfinite(vec)]
        vec = vec[vec > 0]
        return float(np.percentile(vec, q)) if vec.size else 1.0

    vmax_H0 = vmax_compartido([img_nocaos_H0, img_caos_H0], q = 98)
    vmax_H1 = vmax_compartido([img_nocaos_H1, img_caos_H1], q = 98)

    def preparar_vista(im, vmax, gamma = 0.35, eps = 1e-9):
        im = np.nan_to_num(im, nan = 0.0, posinf = 0.0, neginf = 0.0)
        im = im / max(vmax, eps)
        im = np.clip(im, 0.0, 1.0)
        im = np.power(im + 1e-6, gamma)
        return im

    vista_nocaos_H0 = preparar_vista(img_nocaos_H0, vmax_H0, gamma = 0.35)
    vista_nocaos_H1 = preparar_vista(img_nocaos_H1, vmax_H1, gamma = 0.35)
    vista_caos_H0 = preparar_vista(img_caos_H0, vmax_H0, gamma = 0.35)
    vista_caos_H1 = preparar_vista(img_caos_H1, vmax_H1, gamma = 0.35)

    fig, axs = plt.subplots(2, 2, figsize = (8, 8))
    axs[0, 0].imshow(vista_nocaos_H0, cmap = "inferno", origin = "lower", vmin = 0, vmax = 1)
    axs[0, 0].set_title(f"No caótica - $H_0$ (d={dim_pi}, τ={retardo_pi})")
    axs[0, 1].imshow(vista_nocaos_H1, cmap = "inferno", origin = "lower", vmin = 0, vmax = 1)
    axs[0, 1].set_title(f"No caótica - $H_1$ (d={dim_pi}, τ={retardo_pi})")
    axs[1, 0].imshow(vista_caos_H0, cmap = "inferno", origin = "lower", vmin = 0, vmax = 1)
    axs[1, 0].set_title(f"Caótica - $H_0$ (d={dim_pi}, τ={retardo_pi})")
    axs[1, 1].imshow(vista_caos_H1, cmap = "inferno", origin = "lower", vmin = 0, vmax = 1)
    axs[1, 1].set_title(f"Caótica - $H_1$ (d={dim_pi}, τ={retardo_pi})")
    for ax in axs.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    principal()
    # experimento_diferentes_semillas(n_ejecuciones = 5)
    # ilustracion()



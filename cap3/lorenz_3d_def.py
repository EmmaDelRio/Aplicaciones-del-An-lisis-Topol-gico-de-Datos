"""
APLICACIONES DEL TDA - DETECCIÓN ÓRBITAS CAÓTICAS 
SISTEMA DE LORENZ: nube 3d (castellano)
"""
import numpy as np                          
import matplotlib.pyplot as plt  
import pandas as pd 
from ripser import ripser
from persim import plot_diagrams
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

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
                print("no coincide, se omite")
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

def embed_directo(dataset, mantener_ultimos = None, submuestreo = 1, normalizar = True):
    # Prepara la nube de puntos directamente desde la serie (x, y, z)
    
    for item in dataset:
        S = np.asarray(item['serie'], dtype = np.float64)  # (T, 3)
        if mantener_ultimos is not None and mantener_ultimos > 0 and len(S) > mantener_ultimos:
            S = S[-mantener_ultimos:]
        if submuestreo and submuestreo > 1:
            S = S[::submuestreo]
        if normalizar:
            S = MinMaxScaler().fit_transform(S)
        item['embedded'] = S
    return dataset

# APLICAR TDA

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
    # Convierte una lista de diagramas en diccionario {dim: diagrama}
    
    if isinstance(diagrama, list):
        return {i: diagrama[i] for i in range(len(diagrama))}
    return dict(diagrama)

def nacimiento_persistencia(puntos):
    # Calcula pares (nacimiento, persistencia)
    
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

# DEFINICIÓN Y ENTRENAMIENTO DE LA CNN

def preparar_entrenamiento(dataset):
    # Extrae imágenes y etiquetas para el entrenamiento
    
    X_lista = [item['persimg'] for item in dataset]
    X = np.stack(X_lista, axis = 0)
    y = np.array([item['label'] for item in dataset])

    print(f"Forma de X: {X.shape}")
    print(f"Forma de y: {y.shape}")
    return X, y

class Tiny2DBackbone(nn.Module):
    def __init__(self, in_channels = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 5, kernel_size = 3, dilation = 2, padding = 2, bias = True)
        self.act1  = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(5, 10, kernel_size = 3, dilation = 4, padding = 4, bias = True)
        self.act2  = nn.ReLU(inplace = True)
        self.gap   = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):           
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.gap(x).view(x.size(0), -1)
        return x

class PersistenceCNN(nn.Module):
    def __init__(self, input_channels = 3, num_classes = 2):
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
        feats = self.backbone(x)
        logits = self.fc(feats)
        return logits

def preparar_cargador_datos(X, y, tamano_lote = 32, mezclar = False):
    # Convierte los arrays X e y en un DataLoader de PyTorch
    
    X_t = torch.tensor(X, dtype = torch.float32).permute(0, 3, 1, 2)
    y_t = torch.tensor(y, dtype = torch.long)
    ds  = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size = tamano_lote, shuffle = mezclar)

def entrenar_cnn(modelo, cargador_entrenamiento, cargador_validacion,
                 epocas = 2000, tasa_aprendizaje = 8e-3, dispositivo = "cpu",
                 paciencia = 100, decaimiento_pesos = 1e-5,
                 pesos_clase = None, verbose = True):
    # Entrena la red CNN con early stopping y devuelve el mejor modelo
    modelo.to(dispositivo)
    optimizador = optim.Adam(modelo.parameters(), lr = tasa_aprendizaje, weight_decay = decaimiento_pesos)
    criterio = nn.CrossEntropyLoss(weight = (pesos_clase.to(dispositivo) if pesos_clase is not None else None))

    mejor_perdida_val = float("inf")
    contador_paciencia = 0
    mejor_estado = None

    historial = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoca in range(epocas):
        # Entrenamiento
        modelo.train()
        perdida_total, aciertos, total = 0.0, 0, 0

        for X_batch, y_batch in cargador_entrenamiento:
            X_batch, y_batch = X_batch.to(dispositivo), y_batch.to(dispositivo)

            optimizador.zero_grad()
            salidas = modelo(X_batch)
            perdida = criterio(salidas, y_batch)
            perdida.backward()
            optimizador.step()

            perdida_total += perdida.item() * X_batch.size(0)
            aciertos       += (salidas.argmax(1) == y_batch).sum().item()
            total          += y_batch.size(0)

        perdida_entrenamiento = perdida_total / max(1, total)
        exactitud_entrenamiento = aciertos / max(1, total)

        # Validación
        modelo.eval()
        aciertos_val, total_val = 0, 0
        perdida_val_suma = 0.0

        with torch.no_grad():
            for X_batch, y_batch in cargador_validacion:
                X_batch, y_batch = X_batch.to(dispositivo), y_batch.to(dispositivo)
                salidas = modelo(X_batch)
                perdida = criterio(salidas, y_batch)
                perdida_val_suma += perdida.item() * X_batch.size(0)
                aciertos_val     += (salidas.argmax(1) == y_batch).sum().item()
                total_val        += y_batch.size(0)

        perdida_validacion = perdida_val_suma / max(1, total_val)
        exactitud_validacion = aciertos_val / max(1, total_val)

        # Guardar métricas
        historial["train_loss"].append(perdida_entrenamiento)
        historial["val_loss"].append(perdida_validacion)
        historial["train_acc"].append(exactitud_entrenamiento)
        historial["val_acc"].append(exactitud_validacion)

        # Mostrar progreso
        if verbose and (epoca % 10 == 0 or epoca == epocas - 1):
            print(f"Época {epoca+1:4d}/{epocas} - "
                  f"Train {perdida_entrenamiento:.4f}/{exactitud_entrenamiento:.3f} - "
                  f"Val {perdida_validacion:.4f}/{exactitud_validacion:.3f}")

        # Early stopping por pérdida de validación
        if perdida_validacion < mejor_perdida_val - 1e-6:
            mejor_perdida_val = perdida_validacion
            contador_paciencia = 0
            mejor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
        else:
            contador_paciencia += 1

        if contador_paciencia >= paciencia:
            if verbose:
                print(f"Early stopping en época {epoca+1} | mejor val_loss: {mejor_perdida_val:.4f}")
            break

    # Cargar mejor modelo
    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
        if verbose:
            print(f"Se cargó el mejor modelo (val_loss={mejor_perdida_val:.4f})")

    return modelo, historial

def graficar_perdida_precision(historial):
    # Plot de training and validation loss and accuracy
    
    epocas = range(1, len(historial["train_loss"]) + 1)

    plt.figure(figsize = (10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epocas, historial["train_loss"], label = "Train Loss")
    plt.plot(epocas, historial["val_loss"], label = "Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Val Loss", fontsize = 20)
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epocas, historial["train_acc"], label = "Train Accuracy")
    plt.plot(epocas, historial["val_acc"], label = "Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train/Val Accuracy", fontsize = 20)
    plt.legend()

    plt.tight_layout()
    plt.show()

def evaluar_modelo(modelo, cargador, dispositivo = "cpu", criterio = None):
    # Evalúa el modelo en validación o test y muestra métricas y gráficas
    modelo.eval()
    etiquetas, predicciones, probabilidades = [], [], []
    perdida_total, total_muestras = 0.0, 0

    with torch.no_grad():
        for X_batch, y_batch in cargador:
            X_batch, y_batch = X_batch.to(dispositivo), y_batch.to(dispositivo)
            salidas = modelo(X_batch)

            if criterio is not None:
                perdida = criterio(salidas, y_batch)
                perdida_total += perdida.item() * X_batch.size(0)
                total_muestras += X_batch.size(0)

            probs = torch.softmax(salidas, dim = 1)
            preds = probs.argmax(dim = 1)

            etiquetas.extend(y_batch.cpu().numpy())
            predicciones.extend(preds.cpu().numpy())
            probabilidades.extend(probs[:, 1].cpu().numpy())

    etiquetas = np.array(etiquetas)
    predicciones = np.array(predicciones)
    probabilidades = np.array(probabilidades)

    perdida_media = perdida_total / total_muestras if (criterio is not None and total_muestras > 0) else None

    # Métricas
    matriz_conf = confusion_matrix(etiquetas, predicciones)
    exactitud_global = (predicciones == etiquetas).mean()
    exactitud_por_clase = np.diag(matriz_conf) / np.maximum(matriz_conf.sum(axis = 1), 1)

    # Heatmap
    plt.figure(figsize = (5, 4))
    sns.heatmap(matriz_conf, annot = True, fmt = "d", cmap = "Blues",
                xticklabels = ["no caótico", "caótico"],
                yticklabels = ["no caótico", "caótico"])
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta real")
    plt.show()

    print("\nClasificación (sklearn):")
    print(classification_report(etiquetas, predicciones, target_names = ["no caótico", "caótico"]))
    print(f"Exactitud global: {exactitud_global:.4f}")
    print(f"Exactitud por clase: no caótico: {exactitud_por_clase[0]:.4f} | caótico: {exactitud_por_clase[1]:.4f}")
    if perdida_media is not None:
        print(f"Pérdida media (Val/Test): {perdida_media:.4f}")

    # Histograma de probabilidades
    plt.figure(figsize = (6, 4))
    plt.hist(probabilidades[etiquetas == 0], bins = 20, alpha = 0.7, label = "No caótico")
    plt.hist(probabilidades[etiquetas == 1], bins = 20, alpha = 0.7, label = "Caótico")
    plt.xlabel("Probabilidad de ser caótico")
    plt.ylabel("Número de muestras")
    plt.legend()
    plt.show()

    return etiquetas, predicciones, probabilidades

# PARA LA INTRO DEL CAP.

def ilustracion():
    # Genera una ilustración comparando series caóticas y no caóticas del sistema de Lorenz
    
    RUTA = "C:/Users/Emma/Desktop/cap3/LS_TRAIN_Data_Paper_norm.txt"
    UMBRAL_LE = 0.01
    T_VIS = 500
    SUBMUESTREO = 2
    
    print("Inicio")
    print("Cargando conjunto de datos")
    
    dataset = cargar_lorenz(RUTA, umbral_le = UMBRAL_LE)

    print("Extrayendo series")
    serie_no_caos = next(np.array(item['serie']) for item in dataset if item['label'] == 0)
    serie_caos    = next(np.array(item['serie']) for item in dataset if item['label'] == 1)
    serie_no_caos = serie_no_caos[-T_VIS:]
    serie_caos    = serie_caos[-T_VIS:]
    
    # Figura 1: series temporales
    print("figura 1")
    plt.figure(figsize = (10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(serie_no_caos[-100:, 0], lw = 1)
    plt.xlabel("Tiempo")
    plt.ylabel("x")
    plt.title("No caótica", fontsize = 20)
    plt.subplot(1, 2, 2)
    plt.plot(serie_caos[-100:, 0], lw = 1, color = 'red')
    plt.xlabel("Tiempo")
    plt.ylabel("x")
    plt.title("Caótica", fontsize = 20)
    plt.tight_layout()
    plt.show()

    # Figura 2: diagramas de persistencia
    print("figura 2")
    def normalizar_puntos(P):
        
        return MinMaxScaler().fit_transform(P)

    P_no_caos = normalizar_puntos(serie_no_caos[::SUBMUESTREO])
    P_caos    = normalizar_puntos(serie_caos[::SUBMUESTREO])

    dgms_no_caos = ripser(P_no_caos, maxdim = 2)['dgms']
    dgms_caos    = ripser(P_caos, maxdim = 2)['dgms']
    
    fig, axs = plt.subplots(1, 2, figsize = (10, 4))
    plot_diagrams(dgms_no_caos, show = False, ax = axs[0])
    axs[0].set_title("No caótica", fontsize = 20)
    plot_diagrams(dgms_caos, show = False, ax = axs[1])
    axs[1].set_title("Caótica", fontsize = 20)
    plt.tight_layout()
    plt.show()   
    
    print("Fin")

# PROGRAMA PRINCIPAL (1 semilla)

def principal(
    ruta_train = "C:/Users/Emma/Desktop/cap3/LS_TRAIN_Data_Paper_norm.txt",
    ruta_val   = "C:/Users/Emma/Desktop/cap3/LS_VALIDATION_Data_Paper_norm.txt",
    ruta_test  = "C:/Users/Emma/Desktop/cap3/LS_TEST_Data_Paper_norm.txt",
    mantener_ultimos = None,          
    submuestreo = 1,             
    sigma_px = 1.0,
    resolucion = 32,
    tamano_lote = 128,
    tasa_aprendizaje = 8e-3,
    decaimiento_pesos = 1e-5,
    paciencia = 100,
    epocas = 2000
):
    # Pipeline completo de carga de datos, persistencia, vectorización, entrenamiento y test
    print("Cargando conjuntos de datos")
    train_dataset = cargar_lorenz(ruta_train, umbral_le = 0.01)
    val_dataset   = cargar_lorenz(ruta_val,   umbral_le = 0.01)
    test_dataset  = cargar_lorenz(ruta_test,  umbral_le = 0.01)

    train_dataset = embed_directo(train_dataset, mantener_ultimos, submuestreo, True)
    val_dataset   = embed_directo(val_dataset,   mantener_ultimos, submuestreo, True)
    test_dataset  = embed_directo(test_dataset,  mantener_ultimos, submuestreo, True)

    print("Calculando diagramas de persistencia")
    train_dataset = calcular_persistencia(train_dataset, dimensiones_homologia = (0, 1))
    val_dataset   = calcular_persistencia(val_dataset,   dimensiones_homologia = (0, 1))
    test_dataset  = calcular_persistencia(test_dataset,  dimensiones_homologia = (0, 1))

    print("Vectorizando imágenes de persistencia")
    rango_nacimiento, rango_persistencia = rangos_globales(train_dataset, dims = (0, 1))
    train_dataset = vectorizar_pi(train_dataset, dims = (0, 1), resolucion = resolucion,
                                  canales_por_dimension = True, sigma_px = sigma_px,
                                  normalizar = False,
                                  rango_nacimiento = rango_nacimiento,
                                  rango_persistencia = rango_persistencia)
    val_dataset   = vectorizar_pi(val_dataset,   dims = (0, 1), resolucion = resolucion,
                                  canales_por_dimension = True, sigma_px = sigma_px,
                                  normalizar = False,
                                  rango_nacimiento = rango_nacimiento,
                                  rango_persistencia = rango_persistencia)
    test_dataset  = vectorizar_pi(test_dataset,  dims = (0, 1), resolucion = resolucion,
                                  canales_por_dimension = True, sigma_px = sigma_px,
                                  normalizar = False,
                                  rango_nacimiento = rango_nacimiento,
                                  rango_persistencia = rango_persistencia)

    print("Preparando datos para entrenamiento")
    X_train, y_train = preparar_entrenamiento(train_dataset)
    X_val,   y_val   = preparar_entrenamiento(val_dataset)
    X_test,  y_test  = preparar_entrenamiento(test_dataset)

    print("Creando cargadores de datos")
    cargador_train = preparar_cargador_datos(X_train, y_train, tamano_lote, mezclar = True)
    cargador_val   = preparar_cargador_datos(X_val,   y_val,   tamano_lote, mezclar = False)
    cargador_test  = preparar_cargador_datos(X_test,  y_test,  tamano_lote, mezclar = False)

    print("\nEntrenando modelo CNN")
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = X_train.shape[-1]  
    modelo = PersistenceCNN(input_channels = in_channels, num_classes = 2)

    modelo, historial = entrenar_cnn(modelo, cargador_train, cargador_val,
                                     epocas = epocas, tasa_aprendizaje = tasa_aprendizaje,
                                     dispositivo = dispositivo, paciencia = paciencia,
                                     decaimiento_pesos = decaimiento_pesos, verbose = True)

    graficar_perdida_precision(historial)

    print("\nEvaluando en conjunto de test")
    criterio = nn.CrossEntropyLoss()
    _ = evaluar_modelo(modelo, cargador_test, dispositivo, criterio)

    torch.save(modelo.state_dict(), 'persistence_cnn_lorenz_directo.pth')
    print("Modelo guardado como 'persistence_cnn_lorenz_directo.pth'")
    print("\nPROGRAMA COMPLETADO (directo, H0–H1)")

# PARA 5 SEMILLAS

def experimento_diferentes_semillas(
    n_ejecuciones = 5, semillas = None,
    ruta_train = "C:/Users/Emma/Desktop/cap3/LS_TRAIN_Data_Paper_norm.txt",
    ruta_val   = "C:/Users/Emma/Desktop/cap3/LS_VALIDATION_Data_Paper_norm.txt",
    ruta_test  = "C:/Users/Emma/Desktop/cap3/LS_TEST_Data_Paper_norm.txt",
    mantener_ultimos = None, submuestreo = 1,
    sigma_px = 1.0, resolucion = 32,
    tamano_lote = 128, tasa_aprendizaje = 8e-3,
    decaimiento_pesos = 1e-5, paciencia = 100, epocas = 2000
):
    # Ejecuta varios entrenamientos con diferentes semillas aleatorias
    if semillas is None:
        semillas = [42, 678, 3002, 43654, 81828][:n_ejecuciones]

    resultados_train, resultados_val, resultados_test = [], [], []
    perdidas_train, perdidas_val, perdidas_test = [], [], []
    acc_clase0, acc_clase1 = [], []

    for ejec in range(1, n_ejecuciones + 1):
        semilla = semillas[ejec - 1]
        print(f"\n EXPERIMENTO {ejec}/{n_ejecuciones} - Semilla {semilla}")
        random.seed(semilla); np.random.seed(semilla)
        torch.manual_seed(semilla); torch.cuda.manual_seed_all(semilla)

        train_dataset = cargar_lorenz(ruta_train, umbral_le = 0.01)
        val_dataset   = cargar_lorenz(ruta_val,   umbral_le = 0.01)
        test_dataset  = cargar_lorenz(ruta_test,  umbral_le = 0.01)

        train_dataset = embed_directo(train_dataset, mantener_ultimos, submuestreo, True)
        val_dataset   = embed_directo(val_dataset,   mantener_ultimos, submuestreo, True)
        test_dataset  = embed_directo(test_dataset,  mantener_ultimos, submuestreo, True)

        train_dataset = calcular_persistencia(train_dataset, dimensiones_homologia = (0, 1))
        val_dataset   = calcular_persistencia(val_dataset,   dimensiones_homologia = (0, 1))
        test_dataset  = calcular_persistencia(test_dataset,  dimensiones_homologia = (0, 1))

        rango_nacimiento, rango_persistencia = rangos_globales(train_dataset, dims = (0, 1))
        train_dataset = vectorizar_pi(train_dataset, dims = (0, 1), resolucion = resolucion,
                                      canales_por_dimension = True, sigma_px = sigma_px,
                                      normalizar = False,
                                      rango_nacimiento = rango_nacimiento,
                                      rango_persistencia = rango_persistencia)
        val_dataset   = vectorizar_pi(val_dataset,   dims = (0, 1), resolucion = resolucion,
                                      canales_por_dimension = True, sigma_px = sigma_px,
                                      normalizar = False,
                                      rango_nacimiento = rango_nacimiento,
                                      rango_persistencia = rango_persistencia)
        test_dataset  = vectorizar_pi(test_dataset,  dims = (0, 1), resolucion = resolucion,
                                      canales_por_dimension = True, sigma_px = sigma_px,
                                      normalizar = False,
                                      rango_nacimiento = rango_nacimiento,
                                      rango_persistencia = rango_persistencia)

        X_train, y_train = preparar_entrenamiento(train_dataset)
        X_val,   y_val   = preparar_entrenamiento(val_dataset)
        X_test,  y_test  = preparar_entrenamiento(test_dataset)

        cargador_train = preparar_cargador_datos(X_train, y_train, tamano_lote, mezclar = True)
        cargador_val   = preparar_cargador_datos(X_val,   y_val,   tamano_lote, mezclar = False)
        cargador_test  = preparar_cargador_datos(X_test,  y_test,  tamano_lote, mezclar = False)

        dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        modelo = PersistenceCNN(input_channels = X_train.shape[-1], num_classes = 2)

        cls_counts = np.bincount(y_train, minlength = 2)
        cls_weights = torch.tensor((cls_counts.max() / np.clip(cls_counts, 1, None)).astype("float32"))

        modelo, historial = entrenar_cnn(modelo, cargador_train, cargador_val,
                                         epocas = epocas, tasa_aprendizaje = tasa_aprendizaje,
                                         dispositivo = dispositivo, paciencia = paciencia,
                                         decaimiento_pesos = decaimiento_pesos,
                                         pesos_clase = cls_weights, verbose = True)

        resultados_train.append(historial["train_acc"][-1])
        resultados_val.append(historial["val_acc"][-1])
        perdidas_train.append(historial["train_loss"][-1])
        perdidas_val.append(historial["val_loss"][-1])

        criterio = nn.CrossEntropyLoss(weight = cls_weights)
        modelo.eval(); total_loss, total_n = 0.0, 0
        predicciones, etiquetas = [], []
        with torch.no_grad():
            for xb, yb in cargador_test:
                xb, yb = xb.to(dispositivo), yb.to(dispositivo)
                logits = modelo(xb)
                loss = criterio(logits, yb)
                total_loss += loss.item() * yb.size(0)
                total_n    += yb.size(0)
                predicciones.extend(logits.argmax(1).cpu().numpy())
                etiquetas.extend(yb.cpu().numpy())

        perdida_test = total_loss / max(1, total_n)
        predicciones  = np.array(predicciones)
        etiquetas     = np.array(etiquetas)
        acc_global = (predicciones == etiquetas).mean()
        matriz_conf = confusion_matrix(etiquetas, predicciones)
        acc_por_clase = np.diag(matriz_conf) / np.maximum(matriz_conf.sum(axis = 1), 1)

        perdidas_test.append(perdida_test)
        resultados_test.append(acc_global)
        acc_clase0.append(float(acc_por_clase[0]))
        acc_clase1.append(float(acc_por_clase[1]))

        print(f"Run {ejec} | Test Acc: {acc_global:.4f} | Clase0: {acc_por_clase[0]:.4f} | Clase1: {acc_por_clase[1]:.4f} | Test Loss: {perdida_test:.4f}")

    def media_std(arr):
        return float(np.mean(arr)), float(np.std(arr))

    print(f"Train Acc : {media_std(resultados_train)[0]:.4f} ± {media_std(resultados_train)[1]:.4f}")
    print(f"Val   Acc : {media_std(resultados_val)[0]:.4f}   ± {media_std(resultados_val)[1]:.4f}")
    print(f"Test  Acc : {media_std(resultados_test)[0]:.4f}  ± {media_std(resultados_test)[1]:.4f}")
    print(f"Train Loss: {media_std(perdidas_train)[0]:.4f} ± {media_std(perdidas_train)[1]:.4f}")
    print(f"Val   Loss: {media_std(perdidas_val)[0]:.4f}   ± {media_std(perdidas_val)[1]:.4f}")
    print(f"Test  Loss: {media_std(perdidas_test)[0]:.4f}  ± {media_std(perdidas_test)[1]:.4f}")
    print("\nRESULTADOS POR CLASES (TEST)")
    print(f"Clase 0 Acc: {media_std(acc_clase0)[0]:.4f} ± {media_std(acc_clase0)[1]:.4f}")
    print(f"Clase 1 Acc: {media_std(acc_clase1)[0]:.4f} ± {media_std(acc_clase1)[1]:.4f}")

if __name__ == "__main__":
    # principal()
    # experimento_diferentes_semillas(n_ejecuciones = 5)
    ilustracion()

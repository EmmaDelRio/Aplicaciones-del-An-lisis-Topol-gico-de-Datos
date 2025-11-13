En este repositorio se encuentran los códigos que corresponden a la implementación en Python de lo presentado en la memoria del TFG Aplicaciones del Análisis Topológico de Datos. 
El capítulo 2 es el que corresponde a la aplicación del algoritmo Mapper al conjunto de electrocardiogramas Icentia 11k.
El capítulo 3, al estudio de la utilidad de la homología persistente combinada con redes neuronales convolucionales para la clasificación comportamiento caótico - no caótico aplicado al sistema de Lorenz ya  la ecuación logística.

Entre los códigos del capítulo 2 se encuentran el del ejemlo sencillo que se usa como introductorio en el capítulo, los que corresponden al preprocesamiento del conjunto Icentia 11k y creación del conjunto de entrada al algoritmo y
los que se han empleado para aplicar las configuraciones A y B, interpretar los resultados y seleccionar el nodo candidato final. El preprocesamiento consiste, en primer lugar, en la división de señales en ventanas de 25 segundos, la aplicación del filtro
de calidad y la extracción de las 15 características. En segundo lugar, en la asignación del número de clústers que se asignará a cada paciente de manera proporcional al número de ventanas válidas. En tercer lugar, en la aplicación del algoritmo K-Means y 
selección de la ventana real más cercana al representante de cada clúster. Finalmente, en la creación del conjunto de entrada a algoritmo Mapper (crear una matriz que contenga todos los representantes de cada paciente). En cuanto a los relativos a las configuraciones
A y B cabe destacar que para la configuración A hay un único archivo que incluye tanto la aplicación del algoritmo como los coloreados por carcaterísticas que se han empleado para la interpretación, mientras que para la configuración B hay 2 archivos:
uno que incluye la aplicación del algoritmo y los primeros coloreados que se realizan y otro que incluye también la variación en el tamaño de nodos y su rodeado según determinadas medidas o características, ademas de la selección del nodo candidato que se 
desarrolla al final del capítulo 2.

Entre los capítulos del capítulo 3 se encuentra el correspondiente a la propuesta del capítulo aplicada a la ecuación logística y los correspondientes al sistema de Lorenz tanto con la integración numérica como con l aplicación del embedding de Takens. La estructura
de los códigos es la misma, realizando las pequeñas adaptaciones necesarias (por ejemplo, en la ecuación logística generamos series mientras que en el sistema de Lorenz leemos los archivos que contienen los conjuntos del artículo con el que se realiza la comparación,
Deep Learning for chaos detection (de R. Barrio, Á. Lozano, A. Mayora-Cebollero, A. Miguel, A. Ortega, S. Serrano y R. Vigara)).



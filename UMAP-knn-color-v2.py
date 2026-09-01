import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from river import datasets, neighbors, metrics, evaluate
from river.datasets import synth
import umap
import matplotlib.pyplot as plt
import array

# datasets, el primero es un sintetizador de 21 dimensiones con 3 clases
dataset = synth.Waveform(seed = 112)

#Inicializamos UMAP
umap_model = umap.UMAP(n_components=2, n_neighbors=20, n_jobs=16, min_dist=0.0)

knn = neighbors.KNNClassifier(n_neighbors=5)
metric = metrics.Accuracy()

# Variables para almacenamiento y gráfico
batch_size = 400
X_buffer = []
y_buffer = []

separaciones = array.array('i', [])
colores = array.array('i', [])
X_colors = array.array('i', [])
cmap = plt.colormaps["viridis"]

primero = True

accuracies = []
steps = []

X_graph = np.empty((0, 2))
Y_graph = []

# Matplot interactivo
plt.ion()
fig, ax = plt.subplots()
scatt = ax.scatter(X_graph, Y_graph)

print("Procesando flujo de datos...")

# Simulando un incremento de datos en tiempo real
for i, (x, y) in enumerate(dataset):
    # Convertir el diccionario de River a vector para UMAP
    x_vec = list(x.values())
    # Añadimos los nuevos datos al buffer
    X_buffer.append(x_vec)
    y_buffer.append(y)
    
    # Cuando completamos un lote, aplicamos UMAP y entrenamos k-NN de River
    if len(X_buffer) == batch_size:

        print("\n Procesando buffer")
        print(i)

        X_array = np.array(X_buffer)
        
        # Proyectar el lote a un espacio de menor dimensión con UMAP, el primer lote es entrenamiento
        if primero:
            X_trans = umap_model.fit_transform(X_array)
            primero = False
        else:
            X_trans = umap_model.transform(X_array)
        
        # Procesar punto por punto en el espacio proyectado (Stream)
        for j in range(len(X_trans)):
            features = {f"dim_{d}": val for d, val in enumerate(X_trans[j])}
            label = y_buffer[j]
            
            # Evaluación pre-secuencial (Predict -> Metric Update -> Learn)
            y_pred = knn.predict_one(features)

            try:
                separaciones.append(int(y_pred))
            except:
                separaciones.append(0)
                                    
            if y_pred is not None:
                metric.update(label, y_pred)
            
            knn.learn_one(features, label)
        
        # Guardar métricas para la visualización
        accuracies.append(metric.get())
        steps.append(i + 1)
        print("Precisión: " + str(metric.get()))
        print("Vectores clasificados: " + str(len(separaciones)))

        norm = Normalize(vmin=0, vmax=2)
        colores = cmap(norm(separaciones))

        X_graph = np.vstack([X_graph, X_trans])

        scatt.set_offsets(X_trans)
        scatt.set_color(colores)
        plt.pause(0.01)
        ax.relim()             # Recalcula los límites según los nuevos puntos
        ax.autoscale_view()    # Ajusta la vista
    
        fig.canvas.draw()       # Fuerza el dibujado del lienzo
        fig.canvas.flush_events() # Procesa eventos pendientes
        plt.pause(0.01) 

        # Limpiar búfer
        X_buffer = []
        y_buffer = []
        separaciones = []



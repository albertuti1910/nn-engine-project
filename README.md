# Motor de Redes Neuronales

Este proyecto presenta la implementación de un motor de redes neuronales de tipo feed-forward (completamente conectado) desde cero, utilizando **Python** y la librería **NumPy** para los cálculos numéricos.

El objetivo es profundizar en los fundamentos teóricos y prácticos del aprendizaje profundo, implementando manualmente los algoritmos clave como *backpropagation*, optimización por gradiente y técnicas de regularización avanzadas. El motor ha sido validado en los datasets IRIS y MNIST, demostrando un rendimiento competitivo y el correcto funcionamiento de todos sus componentes.

## Características Principales

El motor implementado es modular y extensible, e incluye las siguientes funcionalidades:

-   **Redes Densas Configurables:** Creación de redes con un número variable de capas y neuronas.
-   **Capas:**
    -   `Dense`: Capa completamente conectada.
    -   `Dropout`: Capa de regularización para combatir el sobreajuste.
-   **Funciones de Activación:**
    -   `Sigmoid`, `ReLU`, `Softmax`, `Tanh`.
-   **Funciones de Pérdida:**
    -   `CategoricalCrossEntropy` (para clasificación).
    -   `MeanSquaredError` (preparado para regresión).
-   **Optimizadores Avanzados:**
    -   `Adam`.
    -   `SGD` con Momentum.
    -   `RMSprop`.
-   **Técnicas de Regularización:**
    -   `Dropout`.
    -   `Weight Decay` (Regularización L2).
-   **Inicialización de Pesos:**
    -   `Xavier/Glorot`.
    -   `He` (ideal para ReLU).
-   **Planificadores de Tasa de Aprendizaje (Schedulers):**
    -   `StepDecay`, `ExponentialDecay`, `CosineAnnealing`.
-   **Utilidades y Métricas:**
    -   División de datos (`train_val_test_split`).
    -   Codificación `one-hot`.
    -   Normalización de características.
    -   Cálculo de `accuracy` y `confusion_matrix`.
    -   **Validación Cruzada (k-fold)**.

## Estructura del Proyecto

El repositorio está organizado de la siguiente manera para mantener el código modular y fácil de entender:

```
nn-engine-project/
├── data/ # Datos cacheados para IRIS y MNIST
├── notebooks/
│ ├── demo_iris.ipynb # Demostración básica y prueba de concepto
│ └── experiment_mnist.ipynb # Experimentos exhaustivos y análisis
├── src/
│ ├── init.py
│ ├── layers.py # Definición de capas (Dense, Dropout)
│ ├── losses.py # Funciones de pérdida
│ ├── network.py # Clases principales (NeuralNetwork, Trainer) y Validación Cruzada
│ ├── optimizers.py # Algoritmos de optimización
│ ├── schedulers.py # Planificadores de tasa de aprendizaje
│ └── utils.py # Funciones de utilidad (preprocesamiento, métricas)
├── requirements.txt
└── README.md
```

## Instalación y Configuración

Para ejecutar este proyecto:

1.  **Clona el repositorio:**
    ```bash
    git clone https://github.com/albertuti1910/nn-engine-project
    cd nn-engine-project
    ```

2.  **Crea un entorno virtual:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: .\.venv\Scripts\activate
    ```

3.  **Instala las dependencias:**
    El proyecto solo requiere librerías estándar. Se instalan con `pip`.
    ```bash
    pip install -r requirements.txt
    ```
    El archivo `requirements.txt` debería contener:
    ```
    numpy
    matplotlib
    pandas
    jupyter
    scikit-learn
    ```
    También se puede hacer uso del package manager `uv`:
    ```bash
    uv sync
    ```

## Uso Básico

El motor está diseñado para ser intuitivo. A continuación se muestra un ejemplo básico para entrenar una red en datos de ejemplo:

```python
import numpy as np
from src import NeuralNetwork, Trainer, Adam
from src.utils import one_hot_encode

# 1. Datos de ejemplo
X_train = np.random.rand(1000, 784)
y_train = np.random.randint(0, 10, 1000)
y_train_oh = one_hot_encode(y_train, num_classes=10)

# 2. Definir la arquitectura y crear la red
net = NeuralNetwork(
    layer_sizes=,
    activations=['relu', 'softmax'],
    initialization='he'
)

# 3. Configurar el entrenador
optimizer = Adam(learning_rate=0.001)
trainer = Trainer(net, optimizer, 'categorical_crossentropy')

# 4. Entrenar el modelo
history = trainer.train(X_train, y_train_oh, epochs=10, batch_size=32)

# 5. Evaluar
test_loss, test_acc = trainer.evaluate(X_test, y_test_oh)
print(f"Precisión en Test: {test_acc:.4f}")
```

## Demostraciones y Resultados
Los experimentos principales se encuentran en la carpeta `notebooks/`.

1. `demo_iris.ipynb`

Este cuaderno sirve como una prueba de concepto en el dataset IRIS. Demuestra el correcto funcionamiento de los algoritmos de entrenamiento y optimización en un problema sencillo.

- **Resultado:** Se alcanzó una precisión del **95.65%**, superando el objetivo del 90%.

2. `experiment_mnist.ipynb`

Este cuaderno contiene un análisis exhaustivo sobre el dataset MNIST. Se realizan múltiples experimentos para comparar:

- Diferentes optimizadores (Adam, SGD, RMSprop).
- El efecto de la regularización (Dropout y L2).
- El impacto de diferentes profundidades de red.
- El uso de planificadores de tasa de aprendizaje.
- **Resultado Base:** La configuración inicial ya alcanzó un **97.36%**, superando el objetivo del 80%.
- **Mejor Resultado:** La configuración con **Dropout** fue la más efectiva, logrando una precisión final en el conjunto de test del **98.20%**.
- **Validación Cruzada:** Para validar la robustez del mejor modelo, se aplicó validación cruzada de 5 pliegues, obteniendo una precisión media estable de **95.35%** con una desviación estándar muy baja (0.36%).

Estos resultados confirman la correcta implementación y la alta efectividad del motor neuronal.

## Autores
- Alberto Rivero Monzón
- Amai Suárez Navarro
- José Mataix Pérez

Este proyecto fue realizado para la asignatura de "Optimización y Heurística" del Grado en Ingeniería y Ciencia de Datos de la Universidad de Las Palmas de Gran Canaria (ULPGC).
# 👗 Fashion MNIST - Clasificador de Ropa con Redes Neuronales

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.0+-red.svg)](https://keras.io/)
[![Google Colab](https://img.shields.io/badge/Google-Colab-yellow.svg)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Un proyecto educativo de Deep Learning que utiliza redes neuronales densas para clasificar imágenes de ropa del dataset Fashion MNIST. Incluye visualización interactiva con TensorFlow Playground embebido.

![Fashion MNIST Banner](https://img.shields.io/badge/Dataset-Fashion_MNIST-purple)

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Características](#-características)
- [Dataset Fashion MNIST](#-dataset-fashion-mnist)
- [Arquitectura del Modelo](#-arquitectura-del-modelo)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Resultados](#-resultados)
- [Visualización Interactiva](#-visualización-interactiva)
- [Estructura del Código](#-estructura-del-código)
- [Mejoras Futuras](#-mejoras-futuras)
- [Contribuciones](#-contribuciones)
- [Autor](#-autor)
- [Licencia](#-licencia)

---

## 🎯 Descripción del Proyecto

Este proyecto implementa un **clasificador de imágenes de ropa** utilizando redes neuronales artificiales. El modelo está entrenado con el dataset **Fashion MNIST**, que contiene 70,000 imágenes en escala de grises de 10 categorías diferentes de prendas de vestir.

### Objetivos del Proyecto

- 🎓 **Educativo**: Aprender los fundamentos de Deep Learning y redes neuronales
- 🔬 **Práctico**: Implementar un clasificador funcional desde cero
- 📊 **Visual**: Incluir visualizaciones claras del proceso de entrenamiento y resultados
- 🚀 **Accesible**: Ejecutable en Google Colab sin necesidad de instalación local

---

## ✨ Características

### Capacidades del Modelo

- ✅ **Clasificación Multi-Clase**: Reconoce 10 tipos diferentes de ropa
- ✅ **Normalización de Datos**: Preprocesamiento automático de imágenes (0-1)
- ✅ **Regularización**: Implementa Dropout para prevenir overfitting
- ✅ **Validación Cruzada**: División automática de datos (train/validation/test)
- ✅ **Visualización de Resultados**: Muestra predicciones vs etiquetas reales
- ✅ **Guardado de Modelo**: Exporta el modelo entrenado en formato `.h5`

### Características Técnicas

- **Framework**: TensorFlow 2.x + Keras
- **Arquitectura**: Red Neuronal Densa (Fully Connected)
- **Optimizador**: Adam
- **Función de Pérdida**: Sparse Categorical Crossentropy
- **Métrica**: Accuracy (precisión)

---

## 👔 Dataset Fashion MNIST

### Descripción

Fashion MNIST es un dataset de imágenes creado por **Zalando Research** como reemplazo moderno del clásico MNIST de dígitos manuscritos.

### Especificaciones

| Característica | Detalle |
|----------------|---------|
| **Imágenes de Entrenamiento** | 60,000 |
| **Imágenes de Prueba** | 10,000 |
| **Resolución** | 28x28 píxeles |
| **Canales** | 1 (Escala de grises) |
| **Clases** | 10 categorías |

### Categorías de Ropa

```python
0 → T-shirt/top (Camiseta/Top)
1 → Trouser (Pantalón)
2 → Pullover (Suéter)
3 → Dress (Vestido)
4 → Coat (Abrigo)
5 → Sandal (Sandalia)
6 → Shirt (Camisa)
7 → Sneaker (Zapatilla)
8 → Bag (Bolso)
9 → Ankle boot (Botín)
```

---

## 🏗️ Arquitectura del Modelo

### Estructura de la Red Neuronal

```
┌─────────────────────────────────────┐
│  CAPA DE ENTRADA (Flatten)          │
│  Input: 28x28 → Output: 784         │
│  (Aplana la imagen 2D en vector 1D) │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  CAPA OCULTA (Dense)                │
│  128 Neuronas + ReLU                │
│  (Extracción de características)    │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  DROPOUT (Regularización)           │
│  Tasa: 20%                          │
│  (Previene overfitting)             │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  CAPA DE SALIDA (Dense)             │
│  10 Neuronas + Softmax              │
│  (Probabilidad por clase)           │
└─────────────────────────────────────┘
```

### Parámetros del Modelo

| Componente | Configuración |
|------------|---------------|
| **Neuronas Capa 1** | 784 (automático por Flatten) |
| **Neuronas Capa Oculta** | 128 |
| **Activación Capa Oculta** | ReLU |
| **Dropout** | 0.2 (20%) |
| **Neuronas Salida** | 10 |
| **Activación Salida** | Softmax |
| **Total Parámetros** | ~101,770 |

### Hiperparámetros de Entrenamiento

```python
Épocas: 10
Batch Size: 128
Optimizador: Adam
Learning Rate: Default (0.001)
Validación Split: 10%
Función de Pérdida: Sparse Categorical Crossentropy
```

---

## 📦 Requisitos

### Dependencias Principales

```
tensorflow>=2.0.0
numpy>=1.19.0
matplotlib>=3.3.0
keras>=2.4.0
```

### Para Google Colab

✅ **¡Nada que instalar!** Google Colab ya incluye todas las librerías necesarias.

### Para Entorno Local

```bash
pip install tensorflow numpy matplotlib
```

---

## 🚀 Instalación

### Opción 1: Google Colab (Recomendado)

1. **Abre Google Colab**: [https://colab.research.google.com/](https://colab.research.google.com/)
2. **Crea un nuevo notebook**
3. **Copia y pega el código** del archivo `Mi_primerared_neuronal.ipynb`
4. **Ejecuta las celdas** secuencialmente
5. ✅ ¡Listo para usar!

**Ventajas de Colab:**
- ☁️ No requiere instalación
- 🖥️ GPU gratuita disponible
- 💾 Almacenamiento en Google Drive
- 🔄 Actualizaciones automáticas de librerías

### Opción 2: Jupyter Notebook Local

```bash
# 1. Clonar o descargar el repositorio
git clone https://github.com/dovalless/mi-primer-red-neuronal.git
cd mi-primer-red-neuronal

# 2. Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Iniciar Jupyter Notebook
jupyter notebook

# 5. Abrir Mi_primerared_neuronal.ipynb
```

---

## 💻 Uso

### Ejecución Rápida

```python
# El notebook se ejecuta celda por celda. Aquí un resumen:

# 1. Cargar dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# 2. Preprocesar datos
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 3. Crear modelo
model = keras.Sequential([...])

# 4. Compilar
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5. Entrenar
history = model.fit(x_train, y_train, epochs=10, validation_split=0.1)

# 6. Evaluar
test_loss, test_acc = model.evaluate(x_test, y_test)

# 7. Predecir
predictions = model.predict(x_test)
```

### Personalización del Entrenamiento

```python
# Modificar épocas para más precisión
history = model.fit(x_train, y_train, 
                    epochs=20,  # Aumentar de 10 a 20
                    batch_size=128,
                    validation_split=0.1)

# Ajustar batch size
history = model.fit(x_train, y_train, 
                    epochs=10,
                    batch_size=64,  # Reducir de 128 a 64
                    validation_split=0.1)
```

### Guardar y Cargar Modelo

```python
# Guardar modelo entrenado
model.save('mi_modelo_fashion.h5')

# Cargar modelo previamente entrenado
from tensorflow import keras
model_cargado = keras.models.load_model('mi_modelo_fashion.h5')

# Usar modelo cargado para predicciones
predicciones = model_cargado.predict(x_test[:5])
```

---

## 📊 Resultados

### Rendimiento Típico del Modelo

Después de **10 épocas** de entrenamiento:

| Métrica | Entrenamiento | Validación | Prueba |
|---------|--------------|------------|--------|
| **Accuracy** | ~89-91% | ~87-89% | ~87-88% |
| **Loss** | ~0.29-0.31 | ~0.31-0.35 | ~0.33-0.37 |

### Ejemplo de Salida del Entrenamiento

```
Epoch 1/10
422/422 - 5s - loss: 0.6240 - accuracy: 0.7844 - val_loss: 0.4451 - val_accuracy: 0.8432
Epoch 2/10
422/422 - 2s - loss: 0.4357 - accuracy: 0.8469 - val_loss: 0.3879 - val_accuracy: 0.8593
...
Epoch 10/10
422/422 - 2s - loss: 0.2920 - accuracy: 0.8935 - val_loss: 0.3073 - val_accuracy: 0.8860

Test accuracy: 0.8797, Test loss: 0.3377
```

### Visualización de Resultados

El notebook genera automáticamente:

1. **Muestra de Imágenes de Entrenamiento**: 6 ejemplos con etiquetas
2. **Predicciones vs Realidad**: 8 ejemplos mostrando predicción del modelo vs etiqueta real
3. **Matriz de Confusión** (opcional): Análisis detallado de errores por clase

---

## 🎮 Visualización Interactiva

### TensorFlow Playground Embebido

El notebook incluye una **visualización interactiva** usando TensorFlow Playground:

```python
from IPython.display import IFrame

IFrame('https://playground.tensorflow.org/#activation=relu&...', 
       width=1100, height=700)
```

#### ¿Qué puedes hacer?

- 🔄 **Experimentar con diferentes arquitecturas** de red
- 📈 **Visualizar fronteras de decisión** en tiempo real
- ⚙️ **Ajustar hiperparámetros** interactivamente
- 🎯 **Entender el aprendizaje** de forma visual

#### Parámetros Configurables

- Función de activación (ReLU, Tanh, Sigmoid, Linear)
- Número de capas ocultas
- Neuronas por capa
- Learning rate
- Regularización (L1, L2)
- Batch size
- Dataset de prueba (círculo, espiral, XOR, etc.)

---

## 🗂️ Estructura del Código

### Organización del Notebook

```
Mi_primerared_neuronal.ipynb
│
├── 📌 Sección 1: Configuración Inicial
│   ├── Verificación de GPU
│   └── Importación de librerías
│
├── 📌 Sección 2: Carga y Exploración de Datos
│   ├── Cargar Fashion MNIST
│   ├── Visualizar muestras
│   └── Análisis de dimensiones
│
├── 📌 Sección 3: Preprocesamiento
│   ├── Normalización (0-1)
│   └── Preparación de datos
│
├── 📌 Sección 4: Construcción del Modelo
│   ├── Definición de arquitectura
│   ├── Compilación
│   └── Resumen del modelo
│
├── 📌 Sección 5: Entrenamiento
│   ├── Fit del modelo
│   └── Validación
│
├── 📌 Sección 6: Evaluación
│   ├── Test accuracy/loss
│   └── Predicciones
│
├── 📌 Sección 7: Visualización de Resultados
│   ├── Gráficos de predicciones
│   └── Análisis de errores
│
├── 📌 Sección 8: Guardado del Modelo
│   └── Exportar .h5
│
└── 📌 Sección 9: TensorFlow Playground
    └── Visualización interactiva
```

---

## 🔮 Mejoras Futuras

### Optimizaciones Planificadas

- [ ] **Arquitecturas Avanzadas**
  - Implementar CNN (Redes Convolucionales)
  - Probar arquitecturas pre-entrenadas (Transfer Learning)
  - Experimentar con ResNet, VGG, MobileNet

- [ ] **Técnicas de Regularización**
  - Batch Normalization
  - Data Augmentation
  - Early Stopping
  - Learning Rate Scheduling

- [ ] **Análisis y Métricas**
  - Matriz de confusión completa
  - Precision, Recall, F1-Score por clase
  - Curvas ROC y AUC
  - Visualización de activaciones

- [ ] **Interfaz de Usuario**
  - Web app con Streamlit
  - API REST con FastAPI
  - Aplicación móvil

- [ ] **Despliegue**
  - Dockerización del modelo
  - Despliegue en TensorFlow Serving
  - Integración con cloud (AWS/GCP/Azure)

---

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Si deseas mejorar este proyecto:

### Cómo Contribuir

1. **Fork** el repositorio
2. Crea una **rama** para tu feature (`git checkout -b feature/MejoraMagica`)
3. **Commit** tus cambios (`git commit -m 'Añade MejoraMagica'`)
4. **Push** a la rama (`git push origin feature/MejoraMagica`)
5. Abre un **Pull Request**

### Ideas de Contribución

- 🐛 Reportar bugs
- 💡 Proponer nuevas features
- 📝 Mejorar documentación
- 🎨 Añadir visualizaciones
- 🧪 Crear tests unitarios
- 🌐 Traducir a otros idiomas

---

## 👨‍💻 Autor

**Darwin Manuel Ovalles Cesar**

<p align="left">
<a href="https://www.linkedin.com/in/darwin-manuel-ovalles-cesar-dev" target="_blank">
<img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="LinkedIn - Darwin Ovalles" height="30" width="40" />
</a>
</p>

- 💼 **LinkedIn**: [darwin-manuel-ovalles-cesar-dev](https://www.linkedin.com/in/darwin-manuel-ovalles-cesar-dev)
- 🌐 **GitHub**: [@dovalless](https://github.com/dovalless)
- 📧 **Email**: [Contacto disponible en LinkedIn]

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para más detalles.

```
MIT License

Copyright (c) 2025 Darwin Manuel Ovalles Cesar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

---

## 🙏 Agradecimientos

- **Zalando Research** - Por crear y mantener el dataset Fashion MNIST
- **TensorFlow Team** - Por el excelente framework de Deep Learning
- **Google Colab** - Por proporcionar GPUs gratuitas para entrenamiento
- **Comunidad Open Source** - Por inspiración y conocimiento compartido

---

## 📚 Referencias y Recursos

### Documentación Oficial

- [TensorFlow Documentation](https://www.tensorflow.org/tutorials)
- [Keras Guide](https://keras.io/guides/)
- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [TensorFlow Playground](https://playground.tensorflow.org/)

### Tutoriales Recomendados

- [Deep Learning Specialization - Coursera](https://www.coursera.org/specializations/deep-learning)
- [Fast.ai - Practical Deep Learning](https://course.fast.ai/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials/keras/classification)

### Papers Relacionados

- **Fashion-MNIST**: A Novel Image Dataset for Benchmarking Machine Learning Algorithms
  - [arXiv:1708.07747](https://arxiv.org/abs/1708.07747)

---

<div align="center">

**⭐ Si este proyecto te resultó útil, considera darle una estrella en GitHub ⭐**

**🚀 ¡Feliz Deep Learning! 🚀**

---

Hecho con ❤️ y ☕ por [Darwin Ovalles](https://www.linkedin.com/in/darwin-manuel-ovalles-cesar-dev)

</div>

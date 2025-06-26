import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# Cargar y procesar el dataset desde GitHub
def load_data():
    url = 'https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv'
    data = pd.read_csv(url)

    # Seleccionar características (X) y etiquetas (y)
    X = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1).values
    y = data['diagnosis'].map({'B': 0, 'M': 1}).values  # Convertir a binario

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=2529)

    # Escalar características
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# Construir el modelo de red neuronal
def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Graficar precisión o accuracy
def plot_accuracy(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_plot.png')
    plt.close()

# Ejecutar el modelo
if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    model = build_model(X_train.shape[1])
    history = model.fit(X_train, y_train, epochs=20, batch_size=150, validation_split=0.2)

    # Guardar gráfico de precisión
    plot_accuracy(history)

    # Evaluar el modelo (muestra la matriz de confusión)
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusión:")
    print(cm)

    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

# Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from linear_model import LinearModel



# *** EMPEZAR CÓDIGO AQUÍ ***

def p05a(lr,eps,max_iter,train_path, eval_path, pred_path,seed=42):
    # Datos
    cols = pd.read_csv('./tp1_prog_AM_Bengoechea_Gonzalo/data/mnist_train.csv', nrows=1).columns
    df = pd.read_csv('./tp1_prog_AM_Bengoechea_Gonzalo/data/mnist_train.csv', usecols=cols[1:])
    df_y= pd.read_csv('./tp1_prog_AM_Bengoechea_Gonzalo/data/mnist_train.csv', usecols=["label"])
    y = df_y["label"]

    # Descomentar para ver la imágen 1 o cambiar dicho número para ver otras
    #plt.imshow(np.array(df.iloc[1, :]).reshape(28,28),cmap="gray")
    #plt.colorbar()
    #plt.show()

    # Imágenes que contengan 3 quedan con label 1 y el resto 0
    #y = df_y[0]
    for i in range(len(y)):
        if y[i] != 1:
            y[i] = 0
        else:
            y[i] = 1

    # Equilibrar cantidad de labels para optimizar el clasificador

    #Muestras de 0
    df_0 = df[y == 0]
    df_0 = df_0.sample(n=6000, random_state=seed)

    #Muestras de 1
    df_1 = df[y == 1]
    df_1 = df_1.sample(n=6000, random_state=seed)

    #Concatenar
    X = pd.concat([df_0, df_1])

    #Mismo procedimiento para y
    df_0_y = y[y == 0].sample(n=6000, random_state=seed)
    df_1_y = y[y == 1].sample(n=6000, random_state=seed)
    y = pd.concat([df_0_y, df_1_y])

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42,stratify=y)

    # Se entrena el modelo y se guardan las predicciones
    Modelo=LogisticRegression(step_size=lr, max_iter=max_iter, eps=eps)
    Modelo.fit(x_train,y_train)
    pred=Modelo.predict(x_test)
    np.savetxt(pred_path + "/p05_pred1.txt", pred,delimiter=',')

    pred_copy = pred.copy()
    pred_copy[pred_copy >= 0.5] = 1
    pred_copy[pred_copy < 0.5] = 0

    from sklearn.metrics import accuracy_score
    # Se calcula el accuracy
    accuracy = accuracy_score(y_test, pred_copy)
    print("Accuracy: ", accuracy)


# *** TERMINAR CÓDIGO AQUÍ ***


# Regresión logística con descenso por gradiente

class LogisticRegression(LinearModel):
    """Regresión Logística con Newton como solver.

    Ejemplo de uso:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Corre el método de descenso por gradiente para minimizar J(tita) para regresión logística.

        Args:
            x: ejemplos de entrenamiento (features solamente). Tamaño (m, n).
            y: etiquetas de ejemplos de entrenamiento. Tamaño (m,).
        """
 # *** EMPEZAR CÓDIGO AQUÍ ***
        m, n = x.shape

        # Iniciacion theta en el vector nulo si es None 
        if self.theta is None:
            self.theta = np.zeros(n)

        #Inicialización de parámetros
        def sigmoide(theta):
            return 1 / (1 + np.exp(-theta @ x.T))
        
        def costo(theta):
            val_sigmoide = sigmoide(theta)
            return (-1 / m * (np.dot(y, np.log(val_sigmoide)) + np.dot((1 - y), np.log(1 - val_sigmoide))))
        
        def gradiente(theta):
            val_sigmoide = sigmoide(theta)
            return 1/m * (x.T @ (val_sigmoide - y))
        
        error = 1

        # Descenso por gradiente

        for i in range(self.max_iter):
            grad = gradiente(self.theta)
            nuevo_theta = self.theta - self.step_size * grad
            error = np.linalg.norm(nuevo_theta - self.theta)
            self.theta = nuevo_theta
            self.contador_iteraciones += 1

            if error < self.eps:
                break

            
        # *** TERMINAR CÓDIGO AQUÍ ***

    def predict(self, x):
        """Hace una predicción sobre x nuevos.

        Args:
            x: entradas de tamaño (m, n).

        Returns:
            Salidas de tamaño (m,).
        """
        # *** EMPEZAR CÓDIGO AQUÍ ***
        def proba(theta):
            return 1 / (1 + np.exp(np.dot(-theta, x.T)))
        
        probas = proba(self.theta)
        return probas 
        # *** TERMINAR CÓDIGO AQUÍ ***


# Guardar predicciones para datos de test (p05_pred1.csv)

# cols = pd.read_csv('Data/mnist_test.csv', nrows=1).columns
# df= pd.read_csv('Data/mnist_test.csv', usecols=cols[1:])
# df_y= pd.read_csv('Data/mnist_test.csv', usecols=["label"])

# *** EMPEZAR CÓDIGO AQUÍ ***

# *** TERMINAR CÓDIGO AQUÍ ***

# Entrenar 10 modelos distintos (1 para cada número) con los datos de entrenamiento y
# predecir con ellos la etiqueta de cada imágen en test.
# Guardar las predicciones (p05_predtot.csv).

# *** EMPEZAR CÓDIGO AQUÍ ***
def p05b(lr,eps,max_iter,train_path, eval_path, pred_path,seed=42):
    # Datos
    cols = pd.read_csv('./tp1_prog_AM_Bengoechea_Gonzalo/data/mnist_train.csv', nrows=1).columns
    X_train = pd.read_csv('./tp1_prog_AM_Bengoechea_Gonzalo/data/mnist_train.csv', usecols=cols[1:])
    y_train= pd.read_csv('./tp1_prog_AM_Bengoechea_Gonzalo/data/mnist_train.csv', usecols=["label"])["label"]

    # Descomentar para ver la imágen 1 o cambiar dicho número para ver otras
    #plt.imshow(np.array(df.iloc[1, :]).reshape(28,28),cmap="gray")
    #plt.colorbar()
    #plt.show()

    mod = []
    
    for i in range(10):
        y_copy = y_train.copy()
        for j in range(len(y_copy)):
            if y_copy[j] != i:
                y_copy[j] = 0
            else:
                y_copy[j] = 1
        modelo = LogisticRegression(eps=eps, max_iter=max_iter, step_size=lr)
        print("Entrenando modelo para el número ", i)
        modelo.fit(X_train, y_copy)
        mod.append(modelo)

    # Datos
    cols = pd.read_csv('./tp1_prog_AM_Bengoechea_Gonzalo/data/mnist_test.csv', nrows=1).columns
    X_test = pd.read_csv('./tp1_prog_AM_Bengoechea_Gonzalo/data/mnist_test.csv', usecols=cols[1:])
    y_test = pd.read_csv('./tp1_prog_AM_Bengoechea_Gonzalo/data/mnist_test.csv', usecols=["label"])["label"]

    # Predicciones
    predicciones = []

    for i in range(10):
        predicciones.append(mod[i].predict(X_test))
        np.concatenate(predicciones)
    predicciones_finales = np.array(predicciones).T
    np.savetxt(pred_path + "/p05_predtot.txt", predicciones_finales, delimiter=",")

    predicciones_finales = np.argmax(predicciones_finales, axis=1)

# *** TERMINAR CÓDIGO AQUÍ ***

# En caso que así se quiera, se pueden visualizar los resultados en una matriz de confusión.
# Descomentar en caso afirmativo. Siendo y las labels y pred_final la predicción final.
    from sklearn.metrics import confusion_matrix
    from sklearn import metrics
    confusion_matrix = metrics.confusion_matrix(y_test, predicciones_finales)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)

    cm_display.plot()
    plt.show()

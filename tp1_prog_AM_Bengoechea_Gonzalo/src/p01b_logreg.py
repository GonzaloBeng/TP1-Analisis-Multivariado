import numpy as np
import util
from linear_model import LinearModel
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def p01b(train_path, eval_path, pred_path):
    """Problema 1(b): Regresión Logística con el método de Newton.

    Args:
        train_path: directorio al CSV conteniendo el archivo de entrenamiento.
        eval_path: directorio al CSV conteniendo el archivo de evaluación.
        pred_path: directorio para guardar las predicciones.
    """

    # Se cargan los datos de entrenamiento y de testeo
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_test, y_test = util.load_dataset(eval_path, add_intercept=True)

    # Se entrena el modelo y se guardan las predicciones
    Modelo=LogisticRegression()
    Modelo.fit(x_train,y_train)
    Modelo.graficos(pred_path)
    pred=Modelo.predict(x_test)
    np.savetxt(pred_path + "p01b_logreg.txt", pred,delimiter=',')


class LogisticRegression(LinearModel):
    """Regresión Logística con Newton como solver.

    Ejemplo de uso:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def reglog(self,x,coef):
        """Corre una regresión logística para un conjunto de datos x y devuelve la predicción.

        Args:
            x: Conjunto de datos. Tamaño (m, n).
            coef: Coeficientes de la regresión. Tamaño (m,).

        Returns:
            Salidas de tamaño (m,).
        """

        # *** EMPEZAR CÓDIGO AQUÍ ***
        
        # *** TERMINAR CÓDIGO AQUÍ ***

    def fit(self, x, y, alpha=1):
        """Corre el método de Newton para minimizar J(tita) para regresión logística.

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
        
        def hessiano(theta):
            val_sigmoide = sigmoide(theta)
            return 1/m * (val_sigmoide * (1 - val_sigmoide) * x.T @ x)
        
        error = 1

        # Newton-Raphson

        for i in range(self.max_iter):
            grad = gradiente(self.theta)
            hess = hessiano(self.theta)
            hessiano_inverso = np.linalg.inv(hess)
            nuevo_theta = self.theta - np.dot(hessiano_inverso, grad)
            error = np.linalg.norm(nuevo_theta - self.theta)
            self.theta = nuevo_theta
            self.contador_iteraciones += 1

            if error < self.eps:
                break

            self.coeficientes.append(self.theta)
            self.costo.append(costo(self.theta))

            pred_proba = sigmoide(self.theta)
            pred_proba = pred_proba/alpha
            pred_proba[pred_proba >= 0.5] = 1
            pred_proba[pred_proba < 0.5] = 0
            self.accuracy.append(accuracy_score(y, pred_proba))

            
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

    def graficos(self,pred_path):
        """Crea los siguientes gráficos:
            - Costo vs Iteraciones
            - Accuracy de entrenamiento vs Iteraciones
            - Evolución features (sin graficar el intercept)
        
            Args:
                pred_path: directorio para guardar las imágenes.
        
        """
        # *** EMPEZAR CÓDIGO AQUÍ ***

        plt.clf()
        plt.plot(self.accuracy)
        plt.title("Precisión del modelo")
        plt.xlabel("Iteraciones")
        plt.ylabel("Acuraccy")
        plt.savefig(pred_path + "\grafico.png")

        plt.clf()
        for i in range(1, len(self.coeficientes[0])):
            plt.plot([theta[i] for theta in self.coeficientes])
        plt.title("Coeficientes del modelo")
        plt.xlabel("Iteraciones")
        plt.ylabel("Valor del Coeficiente")
        plt.legend(["Coef 1", "Coef 2"])
        plt.savefig(pred_path + "\grafico_coeficientes.png")

        plt.clf()
        plt.plot(self.costo)
        plt.title("Costo del modelo")
        plt.xlabel("Iteraciones")
        plt.ylabel("Costo")
        plt.savefig(pred_path + "\grafico_de_costo.png")
    
        # *** TERMINAR CÓDIGO AQUÍ ***


import numpy as np
import util
from linear_model import LinearModel
from sklearn.metrics import accuracy_score

def p01e(train_path, eval_path, pred_path, transform = False):
    """Problema 1(e): análisis de discriminante gaussiano (GDA)

    Args:
        train_path: directorio al CSV conteniendo el archivo de entrenamiento.
        eval_path: directorio al CSV conteniendo el archivo de evaluación.
        pred_path: directorio para guardar las predicciones.
    """
    # Se cargan los datos de entrenamiento y de testeo
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_test, y_test = util.load_dataset(eval_path, add_intercept=False)

    # Se entrena el modelo y se guardan las predicciones
    Modelo=GDA()
    Modelo.fit(x_train,y_train, transform)
    pred=Modelo.predict(x_test)
    if transform:
        np.savetxt(pred_path + "\p01h_gda.txt", pred,delimiter=',')
        util.plot(x_train, y_train, Modelo.theta, pred_path + "\plottrain_transform.png")
    else:
        np.savetxt(pred_path + "\p01e_gda.txt", pred,delimiter=',')
        util.plot(x_train, y_train, Modelo.theta, pred_path + "\plottrain.png")


class GDA(LinearModel):
    """Análisis de discriminante gaussiano.

    Ejemplo de uso:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def transformar(self, x):
        return np.sqrt(np.abs(x)) #np.abs(x)
    

    def fit(self, x, y, transform=False):
        """Entrena un modelo GDA.

        Args:
            x: ejemplos de entrenamiento (features solamente). Tamaño (m, n).
            y: etiquetas de ejemplos de entrenamiento. Tamaño (m,).

        Returns:
            theta: parámetros del modelo GDA.
        """

        # *** EMPEZAR CÓDIGO AQUÍ ***

        if transform: 
            x = self.transformar(x)

        m, n = x.shape

        positivos = len(y[y==1])
        negativos = len(y[y==0])
        # print(positivos)
        # print(negativos)

        phi = 1/m * positivos
        #print(phi)

        mu0 = (np.sum(x[y==0], axis=0)) / negativos
        #print(mu0)

        mu1 = (np.sum(x[y==1], axis=0)) / positivos
        #print(mu1)
    
        sigma1 = 1 / m * np.sum((x-mu1)**2, axis=0)
        sigma1 = np.diag(sigma1)
        # print(sigma1)

        # sigma0 = 1 / m * np.sum((x-mu0)**2, axis=0)
        # sigma0 = np.diag(sigma0)

        theta = np.linalg.inv(sigma1) @ (mu1-mu0)
        #print(theta)

        theta0 = 1/2 * (mu0.T @ np.linalg.inv(sigma1) @ mu0 - mu1.T @ np.linalg.inv(sigma1) @ mu1) - np.log((1-phi)/phi)
        # print(theta0)
        
        self.theta = np.append(theta, theta0)
        print(self.theta)    


        # *** TERMINAR CÓDIGO AQUÍ ***

    def predict(self, x):
        """Hace una predicción sobre x nuevos.

        Args:
            x: entradas de tamaño (m, n).

        Returns:
            Salidas de tamaño (m,).
        """
        # *** EMPEZAR CÓDIGO AQUÍ ***
        def sigmoide(z):
            return 1 / (1 + np.exp(-z))
        
        theta = self.theta[:2]
        theta0= self.theta[2]

        z = (theta @ x.T) + theta0

        predicts = sigmoide(z)

        return predicts
        # *** TERMINAR CÓDIGO AQUÍ ***



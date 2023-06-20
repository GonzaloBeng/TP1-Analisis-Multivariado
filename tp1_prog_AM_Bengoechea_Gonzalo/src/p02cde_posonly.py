import numpy as np
import util


from p01b_logreg import LogisticRegression

# Caracter a reemplazar con el sub problema correspondiente.`
WILDCARD = 'X'


def p02cde(train_path, valid_path, test_path, pred_path):
    """Problema 2: regresión logística para positivos incompletos.

    Correr bajo las siguientes condiciones:
        1. en y-labels,
        2. en l-labels,
        3. en l-labels con el factor de correción alfa.

    Args:
        train_path: directorio al CSV conteniendo el archivo de entrenamiento.
        valid_path: directorio al CSV conteniendo el archivo de validación.
        test_path: directorio al CSV conteniendo el archivo de test.
        pred_path: direcotrio para guardar las predicciones.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** EMPEZAR EL CÓDIGO AQUÍ ***

    # Parte (c): Train y test en labels verdaderos.
    # Asegurarse de guardar las salidas en pred_path_c

    x_train, y_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    #Entreno el modelo y obtengo las predicciones
    Model = LogisticRegression()
    Model.fit(x_train, y_train)

    #Grafico
    #Model.graficos(pred_path_d)
    #Predicciones
    predic = Model.predict(x_test)

    #Guardo las predicciones
    np.savetxt(pred_path_c + "\p02c_logreg.txt", predic, delimiter=",")

    util.plot(x_test, y_test, Model.theta, pred_path_c + "\p02c_logreg.png")
    
    
    # Part (d): Train en y-labels y test en labels verdaderos.
    # Asegurarse de guardar las salidas en pred_path_d

    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    #Entreno el modelo y obtengo las predicciones
    Model = LogisticRegression()
    Model.fit(x_train, y_train)

    #Grafico
    #Model.graficos(pred_path_c)
    #Predicciones
    predic = Model.predict(x_test)

    #Guardo las predicciones
    np.savetxt(pred_path_d + "\p02d_logreg.txt", predic, delimiter=",")

    util.plot(x_test, y_test, Model.theta, pred_path_d + "\p02d_logreg.png")

    # Part (e): aplicar el factor de correción usando el conjunto de validación, y test en labels verdaderos.
    # Plot y usar np.savetxt para guardar las salidas en  pred_path_e

    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    x_val, y_val = util.load_dataset(valid_path, label_col='y', add_intercept=True)

    #Entreno el modelo y obtengo las predicciones
    Model = LogisticRegression()
    Model.fit(x_train, y_train)

    #Grafico
    #Model.graficos(pred_path_c)
    #Predicciones
    v_predic = Model.predict(x_val)

    v_pos = np.sum(y_val[y_val == 1])
    p_sum = np.sum(v_predic[y_val == 1])
    alfa = p_sum / v_pos

    Model = LogisticRegression()
    Model.fit(x_train, y_train, alpha=alfa)
    pred = Model.predict(x_test)
    pred = pred/alfa

    #Guardo las predicciones
    np.savetxt(pred_path_e + "\p02e_logreg.txt", pred, delimiter=",")

    util.plot(x_test, y_test, Model.theta, pred_path_e + "\p02e_logreg.png", correction=alfa)
    # *** TERMINAR CÓDIGO AQUÍ
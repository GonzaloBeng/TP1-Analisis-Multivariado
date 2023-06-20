from p01b_logreg import p01b
from p01e_gda import p01e
from p02cde_posonly import p02cde
from p03d_poisson import p03d
# from p05b_lwr import p05b
# from p05c_tau import p05c


correr = 3  #CAMBIAR POR número de problema a correr. 0 corre todos.

# Problema 1
if correr == 0 or correr == 1:
    p01b(train_path='./tp1_prog_AM_Bengoechea_Gonzalo/data/ds1_train.csv',
         eval_path='./tp1_prog_AM_Bengoechea_Gonzalo/data/ds1_valid.csv',
         pred_path='./tp1_prog_AM_Bengoechea_Gonzalo/outputs/p01b/ds1/')

    p01b(train_path='./tp1_prog_AM_Bengoechea_Gonzalo/data/ds2_train.csv',
         eval_path='./tp1_prog_AM_Bengoechea_Gonzalo/data/ds2_valid.csv',
         pred_path='./tp1_prog_AM_Bengoechea_Gonzalo/outputs/p01b/ds2/')

    p01e(train_path='./tp1_prog_AM_Bengoechea_Gonzalo/data/ds1_train.csv',
         eval_path='./tp1_prog_AM_Bengoechea_Gonzalo/data/ds1_valid.csv',
         pred_path='./tp1_prog_AM_Bengoechea_Gonzalo/outputs/p01e/ds1/')

    p01e(train_path='./tp1_prog_AM_Bengoechea_Gonzalo/data/ds2_train.csv',
         eval_path='./tp1_prog_AM_Bengoechea_Gonzalo/data/ds2_valid.csv',
         pred_path='./tp1_prog_AM_Bengoechea_Gonzalo/outputs/p01e/ds2/')
    
    p01e(train_path='./tp1_prog_AM_Bengoechea_Gonzalo/data/ds1_train.csv',
         eval_path='./tp1_prog_AM_Bengoechea_Gonzalo/data/ds1_valid.csv',
         pred_path='./tp1_prog_AM_Bengoechea_Gonzalo/outputs/p01h/ds1', transform=True)

    p01e(train_path='./tp1_prog_AM_Bengoechea_Gonzalo/data/ds2_train.csv',
         eval_path='./tp1_prog_AM_Bengoechea_Gonzalo/data/ds2_valid.csv',
         pred_path='./tp1_prog_AM_Bengoechea_Gonzalo/outputs/p01h/ds2', transform=True)


# Problema 2
if correr == 0 or correr == 2:
    p02cde(train_path='./tp1_prog_AM_Bengoechea_Gonzalo/data/ds3_train.csv',
        valid_path='./tp1_prog_AM_Bengoechea_Gonzalo/data/ds3_valid.csv',
        test_path='./tp1_prog_AM_Bengoechea_Gonzalo/data/ds3_test.csv',
        pred_path='./tp1_prog_AM_Bengoechea_Gonzalo/outputs/p02X')

# Problema 3
if correr == 0 or correr == 3:
    p03d(lr=1e-7,
        train_path='./tp1_prog_AM_Bengoechea_Gonzalo/data/ds4_train.csv',
        eval_path='./tp1_prog_AM_Bengoechea_Gonzalo/data/ds4_valid.csv',
        pred_path='./tp1_prog_AM_Bengoechea_Gonzalo/outputs/p03d')

# Problema 5
if correr == 0 or correr == 5:
    p05b(tau=5e-1,
         train_path='./tp1_prog_AM_Bengoechea_Gonzalo/data/ds5_train.csv',
         eval_path='./tp1_prog_AM_Bengoechea_Gonzalo/data/ds5_valid.csv')

    p05c(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
         train_path='./tp1_prog_AM_Bengoechea_Gonzalo/data/ds5_train.csv',
         valid_path='./tp1_prog_AM_Bengoechea_Gonzalo/data/ds5_valid.csv',
         test_path='./tp1_prog_AM_Bengoechea_Gonzalo/data/ds5_test.csv',
         pred_path='./tp1_prog_AM_Bengoechea_Gonzalo/outputs/p05b/p05c_pred.txt')

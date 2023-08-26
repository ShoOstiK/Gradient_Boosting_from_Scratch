'''Module of class (GradientBoostingRegressor)'''
from typing import Callable, Union, Tuple
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score



class GradientBoostingRegressor:
    '''Class of model (GradientBoosting)'''
    def __init__(
            self,
            n_estimators: int = 100,
            learning_rate: float = 1e-3,
            max_depth: int = 3,
            min_samples_split: int = 2,
            loss: Union[str, Callable] = 'mse',
            verbose: bool = False
    ):
        '''Initialization of model'''
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.params_list = {'max_depth':max_depth,
                           'min_samples_split':min_samples_split
                           }
        self.loss = loss
        self.verbose = verbose
        self.trees_ = []
        self.base_pred_ = None
        self.current_estimation = None



    def _mse(self, y_true: np.array,
                   y_pred: np.array
    ) -> Tuple[float, float]:
        '''Loss funcition (Mean Squared Error)'''
        grad = (y_pred - y_true)
        loss = np.mean((y_pred - y_true) ** 2)
        return loss, grad

    def _mae(self, y_true: np.array,
             y_pred: np.array
             ) -> Tuple[float, float]:
        '''Loss function (Mean Absolute Error)'''
        grad = np.where(y_pred > y_true, 1, np.where(y_pred == y_true, 0, -1))
        loss = np.mean(abs(y_true - y_pred))
        return loss, grad

    def custom_func(self,
                    y_true: np.array,
                    y_pred: np.array,
                    func: Callable
                    ) -> Tuple[float, float]:
        eps = 1e-3
        loss = func(y_true, y_pred)
        grad = (func(y_true, y_pred + eps) - func(y_true, y_pred)) / eps
        return loss, grad

    def fit(self,
            X: np.array,
            y: np.array
            ):
        '''Fitting the model'''
        self.base_pred_ = np.mean(y)
        first_estimation = np.zeros(y.shape[0]) + self.base_pred_
        self.current_estimation = first_estimation.copy()
        for idx in range(self.n_estimators):
            if self.loss == 'mse':
                c_loss, c_grad= self._mse(y, self.current_estimation)
            if self.loss == 'mae':
                c_loss, c_grad = self._mae(y, self.current_estimation)
            elif not(isinstance(self.loss, str)):
                c_loss, c_grad = self.custom_func(y,
                                                  self.current_estimation,
                                                  self.loss
                                                  )
            if self.verbose:
                print(c_loss)
            temp_tree = DecisionTreeRegressor(**self.params_list)
            temp_tree.fit(X, -c_grad)
            self.current_estimation += self.learning_rate * temp_tree.predict(X)
            self.trees_.append(temp_tree)


    def predict(self,
                X: np.array) -> np.array:
        '''Prediction of test targets'''
        predictions = np.zeros(X.shape[0]) + self.base_pred_
        for idx in range(len(self.trees_)):
            predictions += self.learning_rate * self.trees_[idx].predict(X)

        return predictions




import os
import kivy
import numpy as np
from kivy.app import App
from kivy.uix.label import Label
from model import TensorFlowModel
import pandas as pd
import numpy as np
import keras as kr
import tensorflow as tf
class MyApp(App):

    def build(self):
        model = TensorFlowModel()
        model.load(os.path.join(os.getcwd(), 'iris_model.h5'))
        np.random.seed(42)
        iris_data = pd.read_csv('data/iris.csv')
        iris_data.loc[(iris_data['type'] == 
               'Iris-versicolor') & (iris_data['sepal_length'] < 5.0)]
        all_inputs = iris_data[['sepal_length', 'sepal_width',
                             'petal_length', 'petal_width']].values
        all_types = iris_data['type'].values
        print(all_inputs[57]," - ", all_types[57])
        outputs_vals, outputs_ints = np.unique(all_types, return_inverse=True)
        outputs_cats = kr.utils.to_categorical(outputs_ints)
        outputs_vals, outputs_ints = np.unique(all_types, return_inverse=True)
        inds = np.random.permutation(len(all_inputs))
        train_inds, test_inds = np.array_split(inds, 2)
        inputs_train, outputs_train = all_inputs[train_inds], outputs_cats[train_inds]
        inputs_test,  outputs_test  = all_inputs[test_inds],  outputs_cats[test_inds]
        
        x =inputs_train
        outputs_train = model.pred(x)
        # result should be
        # 0.01647118,  1.0278152 , -0.7065112 , -1.0278157 ,  0.12216613,
        # 0.37980393,  0.5839217 , -0.04283606, -0.04240461, -0.58534086
        return Label(text=f'{y}')


if __name__ == '__main__':
    MyApp().run()
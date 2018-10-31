from keras.layers import Dense, Conv1D, Dropout, Flatten, MaxPooling1D, AveragePooling1D, Reshape, GRU, LSTM
from keras.models import save_model, Sequential, load_model
from keras.optimizers import Adam
from datetime import datetime
import numpy as np
import sys
import os

class Q_Model():

    def __init__(self, model_type, state_dim=None, no_of_actions=None, layers=None, hyperparameters=None, path=None):
        self.model_type = model_type
        self.state_dim = state_dim
        self.no_of_actions = no_of_actions
        if model_type == "pretrained":
            self.load(path)
        else:
            self._build(layers, hyperparameters)
            self.loaded_model = False

    def _build(self, layers, hyperparameters):

        model = Sequential()
        for i in range(len(layers)):
            layer = layers[i]

            if i == 0:
                model.add(self._get_layer(layer, input_shape=self.state_dim))
            elif i == len(layers)-1:
                model.add(self._get_layer(layer))
                model.add(self._get_layer(output_shape=self.no_of_actions))
            else:
                model.add(self._get_layer(layer))

        model.compile(loss='mse', optimizer=Adam(lr=hyperparameters['lr']))
        self.model = model
        self.details = self._make_details(layers, hyperparameters)

    def _get_layer(self, layer=None, input_shape=None, output_shape=None):

        if input_shape:
            if layer["type"] == "Dense":
                return Dense(units=layer.get("units", None), input_shape=input_shape, activation=layer.get("activation", "relu"))
            elif layer["type"] == "Reshape":
                return Reshape(target_shape=layer.get("target_shape"), input_shape=input_shape)
            elif layer["type"] == "Conv1D":
                return Conv1D(filters=layer.get("filters", None),
                            kernel_size=layer.get("kernel_size", 1),
                            strides=layer.get("strides", 1),
                            activation=layer.get("activation", None))
            else:
                print("Please select 'Dense' or 'Conv1D' as the first layer of the model.")
                sys.exit()

        elif output_shape:
            return Dense(units=output_shape, activation="linear")

        else:
            if layer["type"] == "Dense":
                return Dense(units=layer.get("units", None), activation=layer.get("activation", "relu"))
            elif layer["type"] == "Conv1D":
                return Conv1D(filters=layer.get("filters", None),
                             kernel_size=layer.get("kernel_size", 1),
                             strides=layer.get("strides", 1),
                             activation=layer.get("activation", None))
            elif layer["type"] == "Dropout":
                return Dropout(rate=layer["rate"])
            elif layer["type"] == "Flatten":
                return Flatten()
            elif layer["type"] == "MaxPooling1D":
                return MaxPooling1D(pool_size=layer.get("pool_size", 2), 
                             strides=layer.get("pool_size", None),
                             padding=layer.get("padding", "valid"))
            elif layer["type"] == "AveragePooling1D":
                return AveragePooling1D(pool_size=layer.get("pool_size", 2), 
                             strides=layer.get("pool_size", None),
                             padding=layer.get("padding", "valid"))
            elif layer["type"] == "GRU":
                return GRU(units=layer.get("units", None), return_sequences=layer.get("return_sequences", False))
            elif layer["type"] == "LSTM":
                return LSTM(units=layer.get("units", None), return_sequences=layer.get("return_sequences", False))
            else:
                print("Unvalid layer. Please select from Dense, Conv1D, Dropout, Flatten, MaxPooling1D, AveragePooling1D.")
                sys.exit()
            
    def _make_details(self, layers, hyperparameters):
        details = ""
        layer_counter = 1
        for layer in layers:
            details += "Layer " + str(layer_counter) + "\n"

            for key in layer.keys():
                details += key + ": " + str(layer[key]) + "\t"

            details += "\n\n"

            layer_counter += 1

        return details

    def fit(self, state, action, q_values):
        q = self.predict(add_dim(state, self.state_dim))[0]
        q[action] = q_values

        self.model.fit(add_dim(state, self.state_dim), add_dim(q, (self.no_of_actions,)), epochs=1, verbose=0)

    def predict(self, state):
        return self.model.predict(add_dim(state, self.state_dim))

    def save(self):
        directory_name = "models/" + self.model_type + " at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(directory_name)
        save_model(self.model, directory_name + "/model.h5")
        if not self.loaded_model:
            with open(directory_name + "/details.txt", 'w') as f:
                f.write(self.details)

    def load(self, path):
        self.loaded_model = True
        self.model = load_model(path)

def add_dim(x, shape):
	return np.reshape(x, (1,) + shape)

if __name__ == '__main__':
    model = Q_Model("Dense", (40,), 3, [{"type":"Dense", "units":30}, {"type":"Dense", "units":30}], {"lr":0.01})
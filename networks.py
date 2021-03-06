from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import matplotlib.pyplot as plt
import build_vcf_models as bm
from keras.layers import LSTM
from keras.layers.embeddings import Embedding

### This script contains functions used in other scripts for the
### easy creation of neural netowrks.

def plain_NN(input_shape, output_shape, n_layers, n_nodes, step_activation = 'relu',
	final_activation = 'sigmoid',optimizer = False, kind_of_model = 'classification', 
	halve_each_layer = False,dropout = False, learning_rate = 0.0001):
	"""  Creates a simple neural network model
	-input_shape = integer that represents the number of features used by the model.
	-output_shape = integer that represents the number of features the model tries to predict.
	-n_layers = the number of layers in the model.
	-n_nodes = the number of nodes in each layer.
	-step_activation = activation function at each step, can be any that keras uses.
	-final_activation = activation function at the final step, can be any that keras uses.
	-optimizers = if provided, it uses the optimizers delivered.
	-halve each layer = if true, each layer has half the nodes as the previous one.
	-dropout = use drouput layers.
	-learning_rate = the learning rate for the model to learn. 
	"""
	model = Sequential()
	model.add(Dense(n_nodes, activation='relu', input_dim=input_shape))
	if halve_each_layer:
		halver = 2
	else:
		halver = 1
	if dropout:
		model.add(Dropout(0.3))
	for i in range(n_layers-1):
		n_nodes = n_nodes // halver
		model.add(Dense(n_nodes,activation = step_activation))
		print(n_nodes)
		if dropout:
			model.add(Dropout(0.3))
	model.add(Dense(output_shape, activation = final_activation))
	if optimizer:
		optimizer = optimizer
	else:
		optimizer = optimizers.RMSprop(lr = learning_rate)
	if kind_of_model == 'classification':
		model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
	if kind_of_model == 'regression':
		model.compile(optimizer = optimizer, loss = 'mse', metrics = ['mae'])
	return model

def simple_autoencoder(data, n_dimensions, epochs = 1000, learning_rate = 0.0001):
	""" Creates a simple autoencoder
	-n_dimensions = number of dimensions to use."""
	encoding_dims = n_dimensions
	input_data = Input(shape=(data.shape[1],))
	encoded = Dense(encoding_dims, activation = 'relu')(input_data)
	decoded = Dense(data.shape[1], activation = 'relu')(encoded)
	autoencoder = Model(input_data, decoded)
	encoder = Model(input_data, encoded)
	encoded_input =  Input(shape = (encoding_dims,))
	sgd = optimizers.SGD(lr = learning_rate)
	autoencoder.compile(optimizer=sgd, loss='mae')
	X_train, y_train, X_CV, y_CV, X_test,y_test = generate_sets(data, data)
	history = autoencoder.fit(X_train.values, X_train.values, epochs = epochs, batch_size = 256, shuffle =  True,
	 validation_data = (X_test.values, X_test.values))
	return encoder, autoencoder, history

def fit_network(model, data, labels, epochs = 100, batch_size = 32,checkpointer = False):
	"""Fits a neural network into a model and returns the history to easily analyze the performance.
	checkpointer: if given a name, creates a checkpointer with that name.
	"""
	X_train, y_train, X_CV, y_CV, X_test,y_test = generate_sets(data, labels)
	if checkpointer:
		model_file = 'saved_models/' + checkpointer
		checkpointer = bm.ModelCheckpoint(filepath= model_file,  
			verbose=0, save_best_only=True)
		history = model.fit(X_train.values, y_train.values, batch_size=batch_size, 
			epochs=epochs,validation_data=(X_CV.values, y_CV.values), verbose=1,
			callbacks = [checkpointer])
		model.load_weights(model_file)
		model.save(model_file)
	else:
		history = model.fit(X_train.values, y_train.values, batch_size=batch_size, 
			epochs=epochs,validation_data=(X_CV.values, y_CV.values), verbose=1)
		model.evaluate(X_test.values, y_test.values)
	return history

def generate_sets(data, labels, norm = False, do_not_split = False):
	"""Generate sets for the training, validating and testing
	norm for normalize the data.
	do_not_split if you want all the data in the same set, but shuffled. 
	"""
	print('generating sets')
	if norm:
		data = bm.normalize(data)
	if do_not_split:
		data = bm.shuffle(data)
		labels = labels.loc[data.index]
		print('sets generated')
		return data, labels
	X_train, X_cvt, y_train, y_cvt = bm.train_test_split(data, labels, train_size = 0.75, random_state = 0)
	X_CV, X_test, y_CV, y_test = bm.train_test_split(X_cvt, y_cvt, train_size = 0.50, random_state = 0)
	print('sets generated')
	return X_train, y_train, X_CV, y_CV, X_test,y_test

def plot_history(history, avoid_loss = True):
	"""Plots the performance of a training process using the history object that the fit_network function returns.
	-avoid_loss = if true, avoids plotting the loss function and only prints the evaluation metric used."""
	metric = ''
	for element in history.history.keys():
		if avoid_loss:
			if 'loss' in element:
				continue
		if element == 'acc':
			plt.plot(history.history[element], label = 'Accuracy')
			metric = 'Accuracy'
		elif element == 'val_acc':
			plt.plot(history.history[element], label = 'Validation Accuracy')
		else:
			plt.plot(history.history[element], label = element)
	plt.legend(fontsize = 20)
	plt.xlabel('Epoch', fontsize = 20)
	plt.xticks(fontsize = 20)
	plt.ylabel(metric, fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.title('Learning process', fontsize = 20)
	plt.show()

def model1_network(input_shape, output_shape):
	"""One pre-made model that works most of the time."""
	model_name = 'delete'
	print('creating NN')
	model = Sequential()
	model.add(Dense(300, activation='relu', input_dim=input_shape))
	model.add(Dropout(0.5))
	model.add(Dense(150, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(60, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(20, activation='relu'))
	model.add(Dropout(0.2))
	if output_shape > 1:
		model.add(Dense(output_shape, activation='softmax'))
	else:
		model.add(Dense(output_shape, activation='sigmoid'))
	model.summary()
	sgd = optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False) #Default optimizer
	model.compile(optimizer = sgd, loss='binary_crossentropy', metrics=['accuracy'])
	return model
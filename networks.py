from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import matplotlib.pyplot as plt
import build_vcf_models as bm
from keras.layers import LSTM
from keras.layers.embeddings import Embedding

def plain_NN(input_shape, output_shape, n_layers, n_nodes, step_activation = 'relu',
	final_activation = 'sigmoid',optimizer = False, kind_of_model = 'classification', 
	halve_each_layer = False,dropout = False):
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
		optimizer = optimizers.RMSprop()
	if kind_of_model == 'classification':
		model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
	if kind_of_model == 'regression':
		model.compile(optimizer = optimizer, loss = 'mse', metrics = ['mae'])
	return model

def simple_autoencoder(data, n_dimensions, epochs = 1000, learning_rate = 0.0001):
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
	model.add(Dense(output_shape, activation='sigmoid'))
	model.summary()
	sgd = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False) #Default optimizer
	model.compile(optimizer = sgd, loss='binary_crossentropy', metrics=['accuracy'])
	return model
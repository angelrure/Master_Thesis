import pandas
from sklearn.preprocessing import normalize
print('reading data')
data = pandas.read_table('general_correct_psis_exons_200.txt', index_col = 0, sep = ',', nrows = 40000) # general_correct_psis_exons_200.txt, muscle_psis_exons_200
conservation = pandas.read_table('conservation_scores_per_position_per_exon_200_40_40_200', sep = '\t', index_col = 0)

##-----
exon_len = 80 # CHANGE EVERY TIME.
intron_len = 200 # CHANGE EVERY TIME.
model_name = 'model5_4_neu'# CHANGE EVERY TIME.
divistion = False # CHANGE EVERY TIME.
only_conservation = False
##-----

### Comment if not working with the correct_psis
data = data.reset_index()
data= data.drop_duplicates(subset = 'index', keep = 'first')
data.index = data['index']
data.drop('index', axis = 1, inplace = True)


data = data[data['Std'] < 0.1]
data = data.drop('Std', 1)
try:
	data = data.drop('col', 1)
except:
	pass

data = data[(data['Average'] > 0.6) | (data['Average'] < 0.4)]
labels = data['Average']
data = data[data['len'] > 80]

print('data managed')

if only_conservation == False:
	if divistion == True:
		print('creating sections')
		data_w = pandas.DataFrame(index = data.index)
		data_w['pree-beg'] = data['1']
		for i in range(2,int(intron_len/2+1)):
			data_w['pree-beg'] += data[str(i)]

		data_w['pree-end'] = data[str(int(intron_len/2)+1)]
		for i in range(int(intron_len/2+2),int(intron_len+1)):
			data_w['pree-end'] += data[str(i)]

		data_w['ex-beg'] = data[str(int(intron_len+1))]
		for i in range(int(intron_len+2),int(intron_len+exon_len/2+1)):
			data_w['ex-beg'] += data[str(i)]

		data_w['ex-end'] = data[str(int(intron_len+exon_len+1))]
		for i in range(int(intron_len+exon_len/2+2),int(exon_len+intron_len+1)):
			data_w['ex-end'] += data[str(i)]

		data_w['post-beg'] = data[str(int(exon_len+intron_len+1))]
		for i in range(int(exon_len+intron_len+2),int(exon_len+intron_len+intron_len/2+1)):
			data_w['post-beg'] += data[str(i)]

		data_w['post-end'] = data[str(int(exon_len+intron_len+exon_len/2+1))]
		for i in range(int(exon_len+intron_len+intron_len/2+2),int(exon_len+intron_len+intron_len+1)):
			data_w['post-end'] += data[str(i)]
		try:
			data_w['len'] = data.len
		except:
			pass

	elif divistion == False:
		print('creating sections')
		data_w = pandas.DataFrame(index = data.index)
		data_w['pree'] = data['1']
		for i in range(2,int(intron_len+1)):
			data_w['pree'] += data[str(i)]
		data_w['ex'] = data[str(int(intron_len+1))]
		for i in range(int(intron_len+2),int(intron_len+exon_len+1)):
			data_w['ex'] += data[str(i)]
		data_w['post'] = data[str(int(exon_len+intron_len+1))]
		for i in range(int(exon_len+intron_len+2),int(exon_len+intron_len+intron_len+1)):
			data_w['post'] += data[str(i)]
		try:
			data_w['len'] = data.len
		except:
			pass

	print('sections created')
	print('creating khmers')
	kmers = {}
	letters = ['A', 'C', 'T', 'G']
	letters_p = {'A':25, 'C':50, 'T':75, 'G':100}

	for i in letters:
		kmers[i] = 0
		for x in letters:
			kmers[i+x] = 0
			for y in letters:
				kmers[i+x+y] = 0
				for k in letters:
					kmers[i+x+y+k] = 0
					for l in letters:
						kmers[i+x+y+k+l] = 0

		print('kmers completed' + str(letters_p[i]))

	print('adding khmers')

	parts = data_w.columns[:-1]
	for part in parts: # for part in ['pree', 'ex-beg','ex-end', 'post']:
		for kmer in kmers:
			a = data_w[part].str.count(kmer)
			if a.sum() >0:
				data_w[part+kmer] = a
			else:
				pass



if only_conservation == True:
	data = data.drop(data.columns[1:], axis = 1)


if only_conservation == False:
	data = data_w[data_w.columns[len(parts):]]

data['Average'] = labels
data = conservation.merge(data, left_index = True, right_index = True)
labels = data['Average']
data = data.drop('Average', 1)

data = data.reindex_axis(sorted(data.columns), axis=1)
data_n = normalize(data)


print('getting subsets of data')
from sklearn.cross_validation import train_test_split
X_train, X_cvt, y_train, y_cvt = train_test_split(data_n, labels, train_size = 0.80)
X_CV, X_test, y_CV, y_test = train_test_split(X_cvt, y_cvt, train_size = 0.20)

#Nueral netowrk
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

model = Sequential()
model.add(Dense(300, activation='relu', input_dim=X_CV.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False) #Default optimizer
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0) #For regression.
#optimizer = keras.optimizers.Adagrad(lr=0.0001, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer, loss='mse', metrics=['mae'])
#model.compile(loss='rmsprop',optimizer=optimizer, metrics=['accuracy']) # For classification
#model.compile(optimizer='rmsprop',loss='mse',metrics=['mae']) #For regression
from keras.callbacks import ModelCheckpoint  
checkpointer = ModelCheckpoint(filepath='saved_models/' + model_name,  verbose=1, save_best_only=True)
model.fit(X_train, y_train, batch_size=300, epochs=2000,validation_data=(X_CV, y_CV), verbose=2, callbacks = [checkpointer])
model.load_weights('saved_models/' + model_name)
model.save('saved_models/' + model_name + 'full')
print(model.evaluate(X_test, y_test))
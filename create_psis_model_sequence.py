import pandas
from sklearn.preprocessing import normalize
print('reading data')
data = pandas.read_table('general_correct_psis_exons_200.txt', index_col = 0, sep = ',', nrows = 40000) # general_correct_psis_exons_200.txt, muscle_psis_exons_200

##-----
exon_len = 80 # CHANGE EVERY TIME.
intron_len = 200 # CHANGE EVERY TIME.
model_name = 'model1_filter_40'# CHANGE EVERY TIME.
divistion = False # CHANGE EVERY TIME.
include_seq = True
##-----

""" # Uncomment if not general_correct_psis
print('data readed')
print('managing data')
cds = pandas.read_table('exons_cds.txt')
inner = pandas.read_table('true_inner.txt', index_col = 0)
data = data.transpose()
c = [c for c in data.columns if c in inner.index]
data = data[c]
cds.index = cds['Exon stable ID']
d = [d for d in data.columns if d in cds.index]
data = data[d]
data = data.transpose()
data = data.dropna()
"""

### Comment if not working with the correct_psis
def manage_duplicates(dataset): #
	dataset = dataset.reset_index()
	dataset = dataset.sort_values('Average')
	dataset = dataset.drop_duplicates(subset = 'index', keep = 'last')
	dataset.index = dataset['index']
	dataset.drop('index', axis = 1, inplace = True)
	return dataset

data = manage_duplicates(data)

data = data[data['Std'] < 0.2]
data = data.drop('Std', 1)
try:
	data = data.drop('col', 1)
except:
	pass

data = data[(data['Average'] > 0.6) | (data['Average'] < 0.4)]
data['Average'] = [round(x) for x in data['Average']]
data = data[data['len'] > 80]
labels = data['Average']
data= data.drop('Average', 1)

print('data managed')

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
#					for m in letters:
#						kmers[i+x+y+k+l+m] = 0
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

print('getting subsets of data')
if include_seq:
	data.drop('len', axis = 1, inplace = True)
	data = pandas.get_dummies(data)
	data = pandas.concat([data, data_w[data_w.columns[len(parts):]]],1)
	data = data.reindex_axis(sorted(data.columns), axis=1)
	data_n = normalize(data)
else:
	data = data_w[data_w.columns[len(parts):]]
	data = data.reindex_axis(sorted(data.columns), axis=1)
	data_n = normalize(data)

from sklearn.cross_validation import train_test_split
X_train, X_cvt, y_train, y_cvt = train_test_split(data_n, labels, train_size = 0.80, random_state = 0)
X_CV, X_test, y_CV, y_test = train_test_split(X_cvt, y_cvt, train_size = 0.50, random_state = 0)

""" #Tree
from sklearn import tree

for i in range (130, 170):
	dtc = tree.DecisionTreeClassifier(max_depth = i)
	dtc.fit(X_train, y_train)
	print(dtc.score(X_train, y_train), dtc.score(X_CV, y_CV))
"""


""" # NN
from sklearn.neighbors import KNeighborsClassifier
for i in range(1, 10):
	neigh = KNeighborsClassifier(n_neighbors=i)
	print(i)
	neigh.fit(X_train, y_train)
	print(neigh.score(X_train, y_train), neigh.score(X_CV, y_CV))
"""

#Nueral netowrk
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

print('creating NN')
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

sgd = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False) #Default optimizer
#optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0) #For regression.
#optimizer = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
model.compile(optimizer = sgd, loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(loss='rmsprop',optimizer=optimizer, metrics=['accuracy']) # For classification
#model.compile(optimizer='rmsprop',loss='mse',metrics=['mae']) #For regression
from keras.callbacks import ModelCheckpoint  
checkpointer = ModelCheckpoint(filepath='saved_models/' + model_name,  verbose=1, save_best_only=True)
history = model.fit(X_train, y_train, batch_size=300, epochs=2000,validation_data=(X_CV, y_CV), verbose=2, callbacks = [checkpointer])
model.load_weights('saved_models/' + model_name)
model.save('saved_models/' + model_name + 'full')
print(model.evaluate(X_test, y_test))
from sklearn.metrics import classification_report
print(classification_report(y_CV, model.predict(X_CV).round()))
print(classification_report(y_test, model.predict(X_test).round()))
import pandas
from scipy.stats import ttest_ind_from_stats
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

def read_sequence(sequence_data, n_samples = 10e10, skip_rows = False, index = 1, discard_mutation_id = True):
	print('Reading sequence data')
	data = pandas.read_table(sequence_data, index_col = index, sep = ',', nrows = n_samples, skiprows = range(1,skip_rows))
	if discard_mutation_id:
		if 'mutation_id' in data.columns:
			data.drop('mutation_id', inplace = True, axis = 1)
	print('Sequence data readed')
	return data

def read_conservation(conservation_data, n_samples = 10e10, skip_rows = False, index = 1, discard_mutation_id = True):
	print('Reading conservation data')
	data = pandas.read_table(conservation_data, nrows = n_samples, index_col = index, skiprows = range(1,skip_rows))
	if discard_mutation_id:
		if 'mutation_id' in data.columns:
			data.drop('mutation_id', inplace = True, axis = 1)
	print('Conservation data readed')
	return data

def read_independent_data(sequence_data = 'general_correct_psis_exons_200_vcf_per_tissue.txt', conservation_data = 'conservation_scores_per_position_per_exon_200_40_40_200_vcf_per_tissue.txt' , n_samples_to_skip = 210000, n_samples = 10e10):
	print('Reading independent data, please choose the same number of samples as in used in the model.')
	data_sequence = pandas.read_table(sequence_data, index_col = 1, sep = ',', skiprows = range(1, n_samples_to_skip), nrows = n_samples)
	data_conservation = pandas.read_table(conservation_data, index_col = 1, skiprows = range(1, n_samples_to_skip), nrows = n_samples)
	return data_sequence, data_conservation

def merge_conservation_and_sequence(sequence_data, conservation_data):
	print('Merging sequence and conservation data')
	if 'len' in sequence_data:
		sequence_data.drop(['len'], axis = 1, inplace = True)
	conservation_data.drop(['Average', 'Std', 'Tissue', 'Number', 'len'], inplace = True, axis = 1, errors = 'ignore')
	conservation_data.columns = conservation_data.columns + 'c'
	try:
		data = pandas.concat([conservation_data, sequence_data], axis = 1)
	except:
		data = sequence_data.join(conservation_data, how = 'inner', rsuffix = 'c') 
	print('Files merged')
	return data

def filter_by_minimum_number_of_tissues(data, minimum_number = 40):
	print('Filtering mutations by having a minimum number of cases across tissues')
	data = data[data['Number'] >= minimum_number]
	print('Filtered')
	return data

def extract_changing_exons(data, bonferrony_correction = True, alpha = 0.01):
	print('Filtering exons by having differences across tissues. This may take a while')
	exons_changing = []
	averages = data[['Average', 'Std','Number']]
	unique_exons = data.index.unique()
	for i, exon in enumerate(unique_exons):
		exon_data = averages.loc[exon]
		len_data = len(exon_data)
		if len(exon_data.shape) <= 1:
			continue
		for j, tissue in enumerate(exon_data.iterrows()):
			tissue_2_generator = exon_data.iterrows()
			for k in range(j+1):
				hide = next(tissue_2_generator)
			for tissue2 in tissue_2_generator:
				p_value = ttest_ind_from_stats(tissue[1].Average, tissue[1].Std, tissue[1].Number, tissue2[1].Average, tissue2[1].Std, tissue2[1].Number)[1]
				if bonferrony_correction: # 3446
					correction = len(exon_data)-j-1
					if p_value < (alpha/correction):
						exons_changing.append(exon)
						break
				else: # 3578
					if p_value < alpha:
						exons_changing.append(exon)
						break
			if exon in exons_changing:
				break
		print('\rPercentage completion: '+str(int(100*i/len(unique_exons))) + '%', end = '')
	print('\nExons obtained, converting data.')
	data = data.loc[exons_changing]
	return data

def calculate_kmers(data, intron_len = 200, exon_len = 80, kmer_dimension = 5, keep_base_sequence = False, verify_nucleotides= True):
	print('Generating kmers')
	data['pree'] = data[[str(x) for x in range(1, intron_len+1)]].sum(1)
	data['ex'] = data[[str(x) for x in range(intron_len + 1, intron_len+exon_len + 1)]].sum(1)
	data['post'] = data[[str(x) for x in range(intron_len + exon_len + 1, intron_len*2+exon_len + 1)]].sum(1)
	kmers = {}
	letters = ['A', 'C', 'T', 'G']
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
	parts = data.columns[-3:]
	for part in parts: # for part in ['pree', 'ex-beg','ex-end', 'post']:
		for kmer in kmers:
			data[part+kmer] = data[part].str.count(kmer)
		print('Part %s completed' %(part))
	if keep_base_sequence == True:
		data.drop(['pree','ex','post'], inplace = True, axis = 1)
		data = pandas.get_dummies(data, columns = [str(x) for x in range(1, 481)])
		if verify_nucleotides == True:
			print('Verifying all positions-nucleotides combinations are considered')
			for position in range(1, 481):
				for letter in ['A','T','C','G']:
					if (str(position) + '_' + letter) not in data.columns:
						data[(str(position) + '_' + letter)] = 0
	else:
		data.drop(['pree','ex','post'] + [str(x) for x in range(1, 481)], inplace = True, axis = 1)
	return data


def prepare_data_for_model(data, kind_of_model, std = 0.2, min_average = 0.3): #kind_of_model in ['Classification', 'Regression']
	if kind_of_model == 'Classification':
		data = data[(data['Average'] > (1-min_average)) | (data['Average'] < min_average)]
		data['Average'] = data['Average'].round()#[round(x) for x in data['Average']]
	if kind_of_model == 'Regression':
		data = data[data['Std'] < std]
	data.drop('Std', axis = 1, inplace = True)
	return data

def manage_tissues(data, validate_tissues = True, filter_by_number = 40):
	data = pandas.get_dummies(data, columns = ['Tissue'])
	if validate_tissues == True:
		tissues = pandas.read_table('general_correct_psis_exons_200_vcf_per_tissue.txt', usecols = ['Tissue'], sep = ',')
		tissues = tissues['Tissue'].unique()
		for tissue in tissues:
			if 'Tissue_'+tissue not in data.columns:
				data['Tissue_'+tissue] = 0
	if filter_by_number != False:
		data = filter_by_minimum_number_of_tissues(data, filter_by_number)
	return data

def generate_sets(data, avoid_overlap = False, use_tissues = True, using_kmers = True, norm = True, validate_tissues = True, 
	train_set_size = 0.8, do_not_split = False, filter_by_number = 40, 
	get_output_in_pandas_format = False, three_classes = False, always_same_split = False):
	print('Generating sets')
	if use_tissues == True:
		data = manage_tissues(data, filter_by_number = filter_by_number)
	if 'Number' in data.columns:
		data.drop('Number', axis = 1, inplace = True)
	data = data.dropna()
	if three_classes:
		data = pandas.get_dummies(data, columns =['Average'])
		for column in ['Average_-1.0', 'Average_0.0', 'Average_1.0']:
			if column not in data.columns:
				data[column] = 0
		labels = data[['Average_-1.0', 'Average_0.0', 'Average_1.0']]
		data.drop(['Average_-1.0', 'Average_0.0', 'Average_1.0'], axis = 1, inplace = True)
	else:
		labels = data['Average']
		data.drop('Average', axis = 1, inplace = True)
	if using_kmers == False:
		data = pandas.get_dummies(data, columns = [str(x) for x in range(1, 481)])
	data = data.reindex_axis(sorted(data.columns), axis=1)
	if avoid_overlap == False:
		if norm:
			data_n = normalize(data)
			if get_output_in_pandas_format == True:
				data_n = pandas.DataFrame(data_n, index = data.index, columns = data.columns)
		else:
			data_n = data
		if do_not_split == False:
			if always_same_split == True:
				X_train, X_cvt, y_train, y_cvt = train_test_split(data_n, labels, train_size = train_set_size, random_state = 0)
				X_CV, X_test, y_CV, y_test = train_test_split(X_cvt, y_cvt, train_size = 0.40, random_state = 0)
			else:
				X_train, X_cvt, y_train, y_cvt = train_test_split(data_n, labels, train_size = train_set_size)
				X_CV, X_test, y_CV, y_test = train_test_split(X_cvt, y_cvt, train_size = 0.40)
		elif do_not_split == True:
			return data_n, labels
	elif avoid_overlap == True:
		exons_train = random.sample(list(data.index.unique()), int(len(data.index.unique())*train_set_size))
		data_train = data.loc[exons_train]
		if three_classes:
			labels_train = labels.loc[exons_train]
		else:
			labels_train = labels[exons_train]
		exons_test = [x for x in data.index.unique() if x not in exons_train]
		data_test = data.loc[exons_test]
		if three_classes:
			labels_test = labels.loc[exons_test]
		else:
			labels_test = labels[exons_test]
		if norm:
			data_train_n = normalize(data_train)
			data_test_n = normalize(data_test)
			if get_output_in_pandas_format == True:
				data_train_n = pandas.DataFrame(data_train_n, index = data_train.index, columns = data_train.columns)
				data_test_n = pandas.DataFrame(data_test_n, index = data_test.index, columns = data_test.columns)
		else:
			data_train_n = data_train
			data_test_n = data_test
		if do_not_split == False:
			X_train, y_train = data_train_n, labels_train
			X_CV, X_test, y_CV, y_test = train_test_split(data_test_n, labels_test, train_size = 0.4)
		elif do_not_split == True:
			print("Possible wrong parameters. Either avoid_overlap is True but you don't need it or do_not_split should be False")
	print('Sets generated')
	return X_train, y_train, X_CV, y_CV, X_test, y_test

def create_NN(shape_X, shape_Y = 0, activation ='sigmoid'):
	print('creating NN')
	model = Sequential()
	model.add(Dense(200, activation='relu', input_dim=shape_X))
	model.add(Dropout(0.5))
	model.add(Dense(50, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(20, activation='sigmoid'))
	model.add(Dropout(0.2))
	if activation == 'sigmoid':
		model.add(Dense(1,activation = 'sigmoid')) 
	elif activation == 'softmax':
		model.add(Dense(shape_Y,activation = 'softmax')) 
	else:
		model.add(Dense(1))
	return model

def create_simple_NN(shape_X, shape_Y, activation ='sigmoid'):
	print('creating simple NN')
	model = Sequential()
	model.add(Dense(50, activation='relu', input_dim=shape_X))
	model.add(Dropout(0.2))
	model.add(Dense(10, activation='sigmoid'))
	model.add(Dropout(0.2))
	if activation == 'sigmoid':
		model.add(Dense(1,activation = 'sigmoid')) 
	elif activation == 'softmax':
		model.add(Dense(shape_Y,activation = 'softmax')) 
	else:
		model.add(Dense(1))
	return model

def single_layer_NN(shape_X, shape_Y, activation ='sigmoid'):
	print('creating simple NN')
	model = Sequential()
	model.add(Dense(shape_Y, activation=activation, input_dim=shape_X))
	return model
	

def create_combined_model(kind_of_model, save_model = False, n_samples = 210000, 
	include_base_expression = False, filter_by_tissue = False, learning_rate = 0.00001, epochs = 10000, keep_base_sequence = False, 
	avoid_overlap = False, get_only_exons_from_base_outside_testing_set = False, 
	use_deltas_instead_of_averages = False, norm = True, relative_deltas = False, 
	use_conservations = True, minimum_number_of_tissues = 40, three_class_model = False):
	global X_train
	a = read_sequence('general_correct_psis_exons_200_vcf_per_tissue_multiple_mutations.txt', n_samples)
	if use_conservations:
		b = read_conservation('conservation_scores_per_position_per_exon_200_40_40_200_vcf_per_tissue_multiple_mutations.txt', n_samples)
	if filter_by_tissue != False:
		a = a[a['Tissue'] == filter_by_tissue]
		if use_conservations:
			b = b[b['Tissue'] == filter_by_tissue]
	a = calculate_kmers(a, keep_base_sequence = keep_base_sequence)
	if use_conservations:
		c = merge_conservation_and_sequence(a,b)
	else:
		c = a
	if minimum_number_of_tissues:
		c = filter_by_minimum_number_of_tissues(c, minimum_number_of_tissues)
	#c = extract_changing_exons(c)
	c = prepare_data_for_model(c, kind_of_model = kind_of_model)
	#del a,b
	if c.shape[0] < 50:
		return
	if use_deltas_instead_of_averages:
		deltas = calculate_delta_psi_mutations(c, kind_of_model = kind_of_model, get_only_psis = True, specific_tissue = False, keep_base_averages = False, relative_deltas = False)
		c['Average'] = deltas['delta_mut']
	X_train, y_train, X_CV, y_CV, X_test, y_test = generate_sets(c, avoid_overlap = avoid_overlap, norm = norm)
	del c
	try:
		if include_base_expression == True and use_deltas_instead_of_averages == False:
			if get_only_exons_from_base_outside_testing_set == False:
				X_train_base, y_train_base = recover_base_expression_of_mutated(specific_tissue = filter_by_tissue, keep_base_sequence = keep_base_sequence, verify_nucleotides = True, kind_of_model = kind_of_model, use_conservations = use_conservations)
			elif get_only_exons_from_base_outside_testing_set == True:
				X_train_base, y_train_base = recover_base_expression_of_mutated(exons_to_use = y_train.index.unique(), specific_tissue = filter_by_tissue, keep_base_sequence = keep_base_sequence, verify_nucleotides = True, kind_of_model = kind_of_model, use_conservations = use_conservations)
			X_train = np.concatenate((X_train, X_train_base), axis = 0)
			y_train = pandas.concat([y_train, y_train_base])
	except:
		print('Base exons not corretly retrieved, probably too few information to retrieve. Aborting tissue')
		return
	if use_deltas_instead_of_averages:
		model = create_NN(X_CV.shape[1], activation = False)
	else:
		model = create_NN(X_CV.shape[1], activation = 'sigmoid')
	if kind_of_model == 'Classification':
		optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, decay=0.0, nesterov=False)
		model.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=['accuracy'])
	elif kind_of_model == 'Regression':
		optimizer = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0) 
		model.compile(optimizer= 'adam', loss='mse', metrics=['mae'])
	if save_model != False:
		checkpointer = ModelCheckpoint(filepath='saved_models/' + save_model,  verbose=0, save_best_only=True)
	else:
		checkpointer = False
	print(model.summary(), model.input_shape)
	history = model.fit(X_train, y_train, batch_size=300, epochs=epochs,validation_data=(X_CV, y_CV), verbose=1, callbacks = [checkpointer])
	if save_model:
		model.load_weights('saved_models/' + save_model)
		model.save('saved_models/' + save_model + 'full')
		test_model(kind_of_model = kind_of_model, tissue = filter_by_tissue, re_calculate_sets = False, X_test = X_test, y_test = y_test, save_results_in_file = False, load_model_from_file = False, model_object =model)
	return history
def build_three_class_model(n_samples, n_samples_to_skip, return_to_test = False):
	model_name = 'model1_vcf_210k_both_kmer_and_conservation_classification_per_tissue_3_classes_d'
	mutated_data = read_sequence('general_correct_psis_exons_200_vcf_per_tissue_multiple_mutations2.0.txt', n_samples, discard_mutation_id = True, skip_rows = n_samples_to_skip)
	conservations_mutated = read_conservation('conservation_scores_per_position_per_exon_200_40_40_200_vcf_per_tissue_multiple_mutations2.0.txt', n_samples, skip_rows = n_samples_to_skip)
	mutated_data = merge_conservation_and_sequence(mutated_data,conservations_mutated)
	mutated_data = mutated_data.reset_index()
	base = pandas.read_table('general_correct_psis_per_tissue_exons_200.txt', index_col = 0, sep = ',', nrows = n_samples)
	base = base[['Average','Std','Tissue']]
	base = base.reset_index()
	data = pandas.merge(mutated_data, base, right_on = ['index', 'Tissue'], left_on = ['Exon', 'Tissue'], how = 'inner')
	data.drop('index', axis = 1, inplace = True)
	data = data[(data['Std_x'] < 0.2) & (data['Std_y'] < 0.2)]
	data['delta'] = data['Average_x'] - data['Average_y']
	data.loc[data['delta'] > 0.2, 'delta'] = 1
	data.loc[data['delta'] < -0.2, 'delta'] = -1
	data.loc[(data['delta'] < 0.2) & (data['delta'] > -0.2), 'delta'] = 0
	data['Average'] = data['delta']
	data_change = data.loc[data['Average'] != 0]
	data = pandas.concat([data[0:(data_change.shape[0]//2)], data_change])
	data.drop(['Average_x','Average_y', 'Std_x','Std_y', 'delta'], axis = 1, inplace = True)
	data= calculate_kmers(data, keep_base_sequence = True)
	data.index = data['Exon']
	data.drop('Exon', inplace = True, axis = 1)
	if return_to_test:
		X_test, y_test = generate_sets(data, get_output_in_pandas_format = True, avoid_overlap = True, norm = True, three_classes = True, filter_by_number = 0, do_not_split = True)
		return
	X_train, y_train, X_CV, y_CV, X_test, y_test = generate_sets(data, get_output_in_pandas_format = True, avoid_overlap = True, norm = True, three_classes = True, filter_by_number = 0)
	model = create_simple_NN(X_CV.shape[1],y_CV.shape[1] ,activation = 'softmax')
	sgd = keras.optimizers.SGD(lr=0.0001, decay=0, momentum=0.9, nesterov=True)
	rms = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=0, decay=0.0)
	adam = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, decay=0.0)
	model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
	checkpointer = ModelCheckpoint(filepath='saved_models/' + model_name,  verbose=0, save_best_only=True)
	model.fit(X_train.values, y_train.values, batch_size=128, epochs=50000,validation_data=(X_CV.values, y_CV.values), verbose=1, callbacks = [checkpointer])
	model.load_weights('saved_models/' + model_name)
	model.evaluate(X_test, y_test.values)
	model.save('saved_models/' + model_name + 'full')
	return model

def build_three_class_model_with_zscore(n_samples, zscore_min = 2, n_samples_to_skip = 0, return_to_test = False):
	model_name = 'model1_vcf_210k_both_kmer_and_conservation_classification_per_tissue_3_classes_zscore_d'
	mutated_data = read_sequence('general_correct_psis_exons_200_vcf_per_tissue_multiple_mutations2.0.txt', n_samples, discard_mutation_id = True, skip_rows = n_samples_to_skip)
	conservations_mutated = read_conservation('conservation_scores_per_position_per_exon_200_40_40_200_vcf_per_tissue_multiple_mutations2.0.txt', n_samples, skip_rows = n_samples_to_skip)
	mutated_data = merge_conservation_and_sequence(mutated_data,conservations_mutated)
	mutated_data = mutated_data.reset_index()
	base = pandas.read_table('general_correct_psis_per_tissue_exons_200.txt', index_col = 0, sep = ',', nrows = n_samples)
	base = base[['Average','Std','Tissue']]
	base = base.reset_index()
	data = pandas.merge(mutated_data, base, right_on = ['index', 'Tissue'], left_on = ['Exon', 'Tissue'], how = 'inner')
	data.drop('index', axis = 1, inplace = True)
	data['zscore'] = (data['Average_x'] - data['Average_y'])/data['Std_y']
	data['delta'] = data['Average_x'] - data['Average_y']
	data.loc[data['zscore'] > zscore_min, 'delta'] = 1
	data.loc[data['zscore'] < -zscore_min, 'delta'] = -1
	data.loc[(data['zscore'] < zscore_min) & (data['zscore'] > -zscore_min), 'delta'] = 0
	data['Average'] = data['delta']
	data_change = data[abs(data['zscore'].abs()) > zscore_min]
	data = pandas.concat([data[0:(data_change.shape[0]//2)], data_change])
	data.drop(['Average_x','Average_y', 'Std_x','Std_y', 'delta', 'zscore'], axis = 1, inplace = True)
	data= calculate_kmers(data, keep_base_sequence = True)
	data.index = data['Exon']
	data.drop('Exon', inplace = True, axis = 1)
	if return_to_test:
		X_test, y_test = generate_sets(data, get_output_in_pandas_format = True, avoid_overlap = True, norm = True, three_classes = True, filter_by_number = 0, do_not_split = True)
		return
	X_train, y_train, X_CV, y_CV, X_test, y_test = generate_sets(data, get_output_in_pandas_format = True, avoid_overlap = True, norm = True, three_classes = True, filter_by_number = 0)
	model = create_simple_NN(X_CV.shape[1],y_CV.shape[1] ,activation = 'softmax')
	sgd = keras.optimizers.SGD(lr=0.0001, decay=0, momentum=0.9, nesterov=True)
	rms = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=0, decay=0.0)
	adam = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, decay=0.0)
	model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
	checkpointer = ModelCheckpoint(filepath='saved_models/' + model_name,  verbose=0, save_best_only=True)
	model.fit(X_train.values, y_train.values, batch_size=128, epochs=50000,validation_data=(X_CV.values, y_CV.values), verbose=1, callbacks = [checkpointer])
	model.load_weights('saved_models/' + model_name)
	model.evaluate(X_test.values, y_test.values)
	model.save('saved_models/' + model_name + 'full')
	return model

def test_three_class_model(n_samples, n_samples_to_skip):
	model_name = 'model1_vcf_210k_both_kmer_and_conservation_classification_per_tissue_3_classes'
	mutated_data = read_sequence('general_correct_psis_exons_200_vcf_per_tissue_multiple_mutations2.0.txt', n_samples, discard_mutation_id = True, skip_rows = n_samples_to_skip)
	conservations_mutated = read_conservation('conservation_scores_per_position_per_exon_200_40_40_200_vcf_per_tissue_multiple_mutations2.0.txt', n_samples, skip_rows = n_samples_to_skip)
	mutated_data = merge_conservation_and_sequence(mutated_data,conservations_mutated)
	mutated_data = mutated_data.reset_index()
	base = pandas.read_table('general_correct_psis_per_tissue_exons_200.txt', index_col = 0, sep = ',', nrows = n_samples)
	base = base[['Average','Std','Tissue']]
	base = base.reset_index()
	data = pandas.merge(mutated_data, base, right_on = ['index', 'Tissue'], left_on = ['Exon', 'Tissue'], how = 'inner')
	data.drop('index', axis = 1, inplace = True)
	data = data[(data['Std_x'] < 0.2) & (data['Std_y'] < 0.2)]
	data['delta'] = data['Average_x'] - data['Average_y']
	data.loc[data['delta'] > 0.2, 'delta'] = 1
	data.loc[data['delta'] < -0.2, 'delta'] = -1
	data.loc[(data['delta'] < 0.2) & (data['delta'] > -0.2), 'delta'] = 0
	data['Average'] = data['delta']
	data_change = data.loc[data['Average'] != 0]
	data = pandas.concat([data[0:(data_change.shape[0]//2)], data_change])
	data.drop(['Average_x','Average_y', 'Std_x','Std_y', 'delta'], axis = 1, inplace = True)
	data= calculate_kmers(data, keep_base_sequence = True)
	data.index = data['Exon']
	data.drop('Exon', inplace = True, axis = 1)
	X_test, y_test = generate_sets(data, get_output_in_pandas_format = True, norm = True, three_classes = True, filter_by_number = 0, do_not_split = True)
	model = keras.models.load_model('saved_models/' + model_name)
	model.evaluate(X_test.values, y_test.values)

def test_model(model_object = False, tissue = False, model_file = 'model1_vcf_210k_both_kmer_and_conservation_classification_per_tissue_3_classes', 
	kind_of_model = 'Classification',
	avoid_overlap = False, n_samples_used = 210000, samples_to_use = 10e10, 
	using_kmers = True, re_calculate_sets = True, X_test = False, y_test = False, 
	save_results_in_file = False, load_model_from_file = True, use_deltas_instead_of_averages = False,
	keep_base_sequence = False, three_classes = False):
	if re_calculate_sets == True:
		if avoid_overlap == True:
			a,b = read_independent_data(n_samples_to_skip = n_samples_used, n_samples = samples_to_use)
		elif avoid_overlap == False:
			a = read_sequence('general_correct_psis_exons_200_vcf_per_tissue_multiple_mutations2.0.txt', skip_rows = n_samples_used, n_samples = samples_to_use)
			b = read_conservation('conservation_scores_per_position_per_exon_200_40_40_200_vcf_per_tissue_multiple_mutations2.0.txt', skip_rows = n_samples_used, n_samples = samples_to_use)
		if using_kmers == True:
			a = calculate_kmers(a, keep_base_sequence = keep_base_sequence, verify_nucleotides = True)
		c = merge_conservation_and_sequence(a,b)
		c = filter_by_minimum_number_of_tissues(c, 40)
		c = prepare_data_for_model(c, kind_of_model = kind_of_model)
		if use_deltas_instead_of_averages:
			deltas = calculate_delta_psi_mutations(c, kind_of_model = kind_of_model, get_only_psis = True, specific_tissue = False, keep_base_averages = False, relative_deltas = False)
			c['Average'] = deltas['delta_mut']
		X_train, y_train, X_CV, y_CV, X_test, y_test = generate_sets(c, False, get_output_in_pandas_format =True, train_set_size = 0.99, three_classes = three_classes)
		X_test = X_train
		y_test = y_train
	if load_model_from_file == True:
		model = keras.models.load_model('saved_models/' +model_file)
	if load_model_from_file == False:
		model = model_object
	X_t = X_test #delte
	y_t = y_test #delete
	print(model.evaluate(X_test, y_test))
	if save_results_in_file != False:
		resuls_file = open(save_results_in_file, 'a')
		resuls_file.write(tissue + '\t' + str(model.evaluate(X_test, y_test, verbose = 0)[1]) + '\n')
		resuls_file.close()
	else:
		print(model.evaluate(X_test, y_test))
	if kind_of_model == 'Regression':
		plot_predictions(model, X_test, y_test)



def plot_predictions(model, X_data, y_data, save_instead = False):
	print('R2 score: ' + str(r2_score(model.predict(X_data), y_data)))
	plt.scatter(model.predict(X_data), y_data, s = 5)
	plt.title(str(r2_score(model.predict(X_data), y_data)))
	if save_instead == False:
		plt.show()
	else:
		plt.savefig('model1_vcf_210k_both_kmer_and_conservation_regression_per_tissue_more40_include_base_on_GTEX.png')

def recover_base_expression_of_mutated(kind_of_model, exons_to_use = False, specific_tissue = False, norm = True, n_samples = 10e10, 
	keep_base_sequence = False, verify_nucleotides = True, get_only_psis = False, use_conservations = True):
	"""This is used to extract the base expression from the exons, that is, the expression from the reference exon without mutations."""
	print('Base expression being included')
	#global base, exons_in_vcf, conservations
	base = pandas.read_table('general_correct_psis_per_tissue_exons_200.txt', index_col = 0, sep = ',', nrows = n_samples)
	exons_in_vcf = read_sequence('general_correct_psis_exons_200_vcf_per_tissue.txt', n_samples = n_samples)
	if use_conservations:
		conservations = pandas.read_table('conservation_scores_per_position_per_exon_200_40_40_200', index_col = 0, nrows = n_samples)
	if specific_tissue:
		print('Filtering by specific tissue: ' + specific_tissue)
		base = base[base['Tissue'] == specific_tissue]
		exons_in_vcf = exons_in_vcf[exons_in_vcf['Tissue'] == specific_tissue]
	if exons_to_use is not False:
		print('Filtering by specific exons, in total: ' + str(len(exons_to_use)))
		base = base.loc[[exon for exon in exons_to_use if exon in base.index]]
		exons_in_vcf = exons_in_vcf.loc[[exon for exon in exons_to_use if exon in exons_in_vcf.index]]
		if use_conservations:
			conservations = conservations.loc[[exon for exon in exons_to_use if exon in conservations.index]]
	base = base[base.index.isin(exons_in_vcf.index)]
	base.index = base.index + '_' + base['Tissue']
	exons_in_vcf.index = exons_in_vcf.index + '_' + exons_in_vcf['Tissue']
	datexons_in_vcf = exons_in_vcf.loc[base.index]
	datexons_in_vcf.index = datexons_in_vcf.index.str.split('_').str[0]
	base.index = base.index.str.split('_').str[0]
	base.drop('len', inplace = True, axis = 1)
	base['Number'] = 0
	if get_only_psis == True:
		if specific_tissue != False:
			return base[['Average', 'Std']]
		else:
			return base[['Average', 'Tissue', 'Std']]
	for element in datexons_in_vcf.index.unique():
		base.loc[element,'Number'] = round(datexons_in_vcf.loc[element, 'Number'].max())
	if use_conservations:
		base = base.join(conservations, how = 'inner', rsuffix = 'c')
	#base.drop('Std', inplace = True, axis = 1)
	base = calculate_kmers(base, keep_base_sequence = keep_base_sequence, verify_nucleotides = verify_nucleotides)
	base = prepare_data_for_model(base, kind_of_model = kind_of_model)
	data, labels = generate_sets(base, False, do_not_split = True, norm = norm)
	# datexons_in_vcf = normalize(data) unnecessary
	print('Base expression generated')
	return data, labels

def test_in_base_GTEX(model = 'model1_vcf_210k_only_kmer_and_sequence_classification_per_tissue_no_number_filter2.0', kind_of_model = 'Classification', n_samples = 10e10, keep_base_sequence_in_kmers = True):
	conservations = read_conservation('conservation_scores_per_position_per_exon_200_40_40_200', index = 0, n_samples = n_samples)
	conservations.columns = conservations.columns + 'c'
	sequence = read_sequence('general_correct_psis_per_tissue_exons_200.txt', index = 0, n_samples = n_samples)
	sequence = sequence[sequence['len'] >= 80]
	temporal = [[],[]] # Creating temporal data because appending directly to a pandas object creates a copy each time and makes the processes abbisamlly longer.
	b = pandas.DataFrame(columns = sequence.columns)
	for row in sequence.iterrows():
		if row[1].name not in temporal[1]:
			temporal[0].append(row[1])
			temporal[1].append(row[1].name)
	b = b.append(temporal[0])
	sequence = sequence[['Average', 'Std','Tissue']]
	del temporal
	b = calculate_kmers(b, verify_nucleotides = True, keep_base_sequence = keep_base_sequence_in_kmers)
	a = sequence.join(conservations)
	a = a.join(b[b.columns[4:]])
	a = prepare_data_for_model(a, kind_of_model = kind_of_model)
	data, labels = generate_sets(a, False, train_set_size = 0.99, filter_by_number =False , do_not_split = True, get_output_in_pandas_format = True)
	model = keras.models.load_model('saved_models/' +model)
	print(model.evaluate(data.values, labels))
	if kind_of_model == 'Regression':
		plot_predictions(model, data.values, labels, save_instead = True)

def calculate_delta_psi_mutations(mutated_data, kind_of_model, get_only_psis = True, specific_tissue = False, keep_base_averages = False, relative_deltas = False):
	"""If get_only_psis is True, the khmers and all the independent variables are ingored, only the base psi are retrieved
	to calculate the delta psi.
	If specific tissue is provided, the dataframe that return the function is already filtered by the tissue and doesn't contain the tissue field. Otherwise it contains it.
	- Relative deltas: instead of substracting deltas, calculate the the % of reduction.
	"""
	#global base_psis, deltas #For testing only
	print('Calculating delta values')
	base_psis = recover_base_expression_of_mutated(exons_to_use = mutated_data.index.unique(), kind_of_model = kind_of_model, get_only_psis = get_only_psis, specific_tissue = specific_tissue)
	deltas = pandas.DataFrame(index = mutated_data.index)
	deltas['Average_mut'] = mutated_data['Average']
	if specific_tissue != False:
		deltas = deltas.join(base_psis)
		return deltas
	if specific_tissue == False:
		base_psis.index = base_psis.index + '_' + base_psis['Tissue']
		base_psis.drop('Tissue', axis = 1, inplace = True)
		deltas['Tissue'] = mutated_data['Tissue']
		deltas.index = deltas.index + '_' + deltas['Tissue']
		deltas.drop('Tissue', axis = 1, inplace = True)
		deltas.reset_index(inplace = True)
		deltas = deltas.join(base_psis, on = 'Tissue')
		deltas.index = deltas['Tissue'].str.split('_').str[0]
		deltas['Tissue'] = deltas['Tissue'].str.split('_').str[1]
	if relative_deltas == False:
		deltas['delta_mut'] = deltas['Average'] - deltas['Average_mut']
	elif relative_deltas == True:
		deltas['delta_mut'] = (deltas['Average_mut'] - deltas['Average'])/deltas['Average']
	if keep_base_averages == False:
		deltas.drop(['Average', 'Average_mut'], axis = 1, inplace = True)
	return deltas

def create_tissue_specific_model(tissue = 'Skin', n_samples = 210000, kind_of_model = 'Regression', learning_rate = 0.0001, epochs = 10000):
	create_combined_model(filter_by_tissue = tissue, n_samples = n_samples, kind_of_model = kind_of_model, save_model = 'model1_vcf_multiplemutations_'+str(n_samples)+'_both_kmer_and_conservation_'+ kind_of_model +'_' +tissue+'_more40_no_overlap_include_trainbase', include_base_expression = True,
		keep_base_sequence = True, learning_rate = learning_rate, avoid_overlap = True, get_only_exons_from_base_outside_testing_set = True, epochs = epochs)

def generate_data_for_all_tissues():
	for tiss in ['Colon', 'Fallopian Tube', 'Ovary', 'Uterus', 'Muscle', 'Testis', 'Spleen',
		'Pituitary', 'Small Intestine', 'Blood Vessel', 'Bladder', 'Salivary Gland',
		'Liver', 'Breast', 'Cervix Uteri', 'Kidney', 'Stomach', 'Thyroid', 'Brain',
		'Nerve', 'Prostate', 'Esophagus', 'Skin', 'Adrenal Gland', 'Lung', 'Vagina',
		'Blood', 'Adipose Tissue', 'Heart', 'Pancreas', '<not provided>']:
		print(tiss)
		create_tissue_specific_model(tissue = tiss, n_samples = 210000, kind_of_model = 'Regression')


def create_only_sequence_model(n_samples = 120000, kind_of_model = 'Classification', filter_by_tissue = False):
	if filter_by_tissue != False:
		tissue = filter_by_tissue
	if filter_by_tissue == False:
		tissue = 'all'
	create_combined_model(filter_by_tissue = filter_by_tissue, n_samples = n_samples, kind_of_model = kind_of_model,  save_model = 'model1_vcf_multiplemutations_'+str(n_samples)+'_only_kmer_'+ kind_of_model +'_' +tissue+'_more40_no_overlap_include_trainbase', include_base_expression = True,
		keep_base_sequence = True, learning_rate = 0.001, avoid_overlap = True, get_only_exons_from_base_outside_testing_set = True, use_conservations = False)


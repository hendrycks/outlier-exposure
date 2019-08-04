import numpy as np
import nltk.data
import os
import csv

train_path = './orig_data/20ng-train-no-short.txt'
test_path = './orig_data/20ng-test-no-short.txt'

class_names = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
			   'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
			   'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
			   'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast',
			   'talk.politics.misc', 'talk.religion.misc']

def reformat(in_path, out_path):
	with open(in_path, 'r') as f:
		lines = f.readlines()

	new_lines = [line.split() for line in lines]
	labels = [line[0] for line in new_lines]  # extract label from line
	labels = [class_names.index(label) for label in labels]  # switch to index
	data = [' '.join(line[1:]) for line in new_lines]  # extract text from line

	with open(out_path, 'w') as myfile:
		wr = csv.writer(myfile)
		for label, inp in zip(labels, data):
			wr.writerow([label, inp])

reformat(train_path, './20ng-train.txt')
reformat(test_path, './20ng-test.txt')
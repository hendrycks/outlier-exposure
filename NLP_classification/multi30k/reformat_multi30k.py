import numpy as np
import nltk.data
import os
import csv


train_path = './orig_data/train.en'
test_path = './orig_data/val.en'

def reformat(in_path, out_path):
	with open(in_path, 'r') as f:
		lines = f.readlines()


	new_lines = []

	for i in range(len(lines)):
		new_line = lines[i].rstrip('\n')
		new_line = " 's ".join(new_line.split("'s"))
		new_line = " 'd ".join(new_line.split("'d"))
		new_line = " ,".join(new_line.split(","))
		new_line = " n't ".join(new_line.split("n't"))
		new_line = ' " '.join(new_line.split('"'))
		if len(new_line) == 0:
			continue
		if new_line[-1] in ['?', '.', '!']:
			new_line = new_line[:-1] + ' ' + new_line[-1]

		new_lines.append(new_line.lower())

	with open(out_path, 'w') as myfile:
		wr = csv.writer(myfile)
		for line in new_lines:
			wr.writerow([line])

reformat(train_path, './train.txt')
reformat(test_path, './val.txt')
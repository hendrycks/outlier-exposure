import numpy as np
import os

wmt16_en_de_path = './wmt16_en_de'

# select the tokenized english test sets
files_to_concat = []
for fname in os.listdir(wmt16_en_de_path):
	if '.tok.en' in fname:
		files_to_concat.append(fname)

# convert each line in these files to lowercase, and concatenate them all together
with open('./wmt16_sentences', 'w') as f_out:
	for fname in files_to_concat:
		with open(os.path.join(wmt16_en_de_path, fname), 'r') as f:
			lines = f.readlines()
			for i, line in enumerate(lines):
				line_out = line.lower()
				f_out.write(line_out)

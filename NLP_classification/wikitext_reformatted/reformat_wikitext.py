import numpy as np
import nltk.data
import os
import csv

# Usage: Uncomment wikitext-2 parts and comment wikitext-103 parts (at top and bottom) to switch.


# wikitext_path = '../wikitext-2/wikitext-2/wiki.train.tokens'
wikitext_path = '../wikitext-103/wikitext-103/wiki.train.tokens'

# Get rid of headers
with open(wikitext_path, 'r') as f:
	lines = f.readlines()

with open('./tmp', 'w') as f:
	for line in lines:
		if (line == ' \n') or (line[:2] == ' ='):
			continue

		f.write(line + '\n')

# Separate sentences with NLTK
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
fp = open('./tmp')
data = fp.read()
sentences = tokenizer.tokenize(data)

# Remove sentences that tend to be poor quality
new_sentences = []
for sentence in sentences:
	if (sentence[0] == '@') or (sentence[0] == ' ') or (sentence[0] == '\n') or (sentence[0] in ['0','1','2','3','4','5','6','7','8','9']):
		continue
	length = len(sentence.split(' '))
	if length > 60 or length < 3:
		continue
	if '\n' in sentence:
		continue
	new_sentences.append(sentence)

os.remove('./tmp')

# Write output

# with open('./wikitext_sentences', 'w') as f:
# 	for sentence in new_sentences:
# 		f.write(sentence + '\n')

with open('./wikitext103_sentences', 'w') as myfile:
	wr = csv.writer(myfile)
	for row in new_sentences:
		wr.writerow([row])


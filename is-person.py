import conceptnet_lite
import nltk
import pandas as pd
from conceptnet_lite import Label, edges_for, edges_between

HUMAN = {'human', 'people', 'person', 'human_adult', 'employee'}
NOUNS = {'NN', 'NNS', 'NNP', 'PRP', 'PRP$'}
HOPS = 2

conceptnet_lite.connect('./conceptnet.db')

def is_word(word):
	try:
		concepts = Label.get(text=word, language='en').concepts
		return True
	except:
		return False

def graph_is_a(word):
	words = set()
	concepts = Label.get(text=word, language='en').concepts
	for e in edges_for(concepts, same_language=True):
		if e.relation.name == 'is_a' and e.start.text == word:
			words.add(e.end.text)
	return words

def next_hop(words):
	next_words = set()
	for word in words:
		if word not in non_human:
			next_words |= graph_is_a(word)
	return next_words

def is_person(word):
	# return true/false if the given term is human
	# true if the term is 2 hops away from any of the human words
	# I_h = {human, people, person, human_adult, employee}, hops = 2
	words = {word}
	for _ in range(HOPS):
		words = next_hop(words)
		if len(words & HUMAN) > 0:
			return True
	return False


twitter_data = pd.read_csv('twitter-all.csv', sep='\t', header=None)
human_file = open('human_words.txt', 'w')
non_human = set()
human = set()
for i, row in twitter_data.iterrows():
	print(i)
	tokens = nltk.word_tokenize(row[2])
	tagged = nltk.pos_tag(tokens)
	for tag in tagged:
		word = tag[0].lower()
		if tag[1] in NOUNS and word not in non_human and word not in human\
			and is_word(word) and is_person(word):
			human.add(word)
			print('HUMAN!!!!', word)
			human_file.write(word + '\n')
		else:
			non_human.add(word)
	human_file.flush()
human_file.close()

# print(is_person('policewoman'))
# print(is_person('terrorist'))

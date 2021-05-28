import conceptnet_lite
import nltk
import pandas as pd
from conceptnet_lite import Label, edges_for, edges_between

NOUNS = {'NN', 'NNS', 'NNP', 'PRP', 'PRP$'}

conceptnet_lite.connect('./conceptnet.db')

# get human words from pre-processed file
human_words_file = open('human_words.txt', 'r')
HUMAN_WORDS = set(human_words_file.read().split('\n'))
human_words_file.close()
print('# human words:', len(HUMAN_WORDS))

def is_word(word):
	try:
		Label.get(text=word, language='en').concepts
		return True
	except:
		return False

def graph_is_a_rev(word):
    words = set()
    concepts = Label.get(text=word, language='en').concepts
    for e in edges_for(concepts, same_language=True):
        if e.relation.name == 'is_a' and e.end.text == word:
            words.add(e.start.text)
    return words

def graph_has_is_a(word1, word2):
    concept1 = Label.get(text=word1, language='en').concepts
    concept2 = Label.get(text=word2, language='en').concepts
    for e in edges_between(concept1, concept2, two_way=False):
        # print(e.start.text, "::", e.end.text, "|", e.relation.name)
        if e.relation.name == 'is_a':
            return True
    return False

Sp = graph_is_a_rev('gender')
print(Sp)

twitter_data = pd.read_csv('twitter-all.csv', sep='\t', header=None)
gendered_file = open('gendered_words.txt', 'w')
non_gendered = set()
gendered = set()
for i, row in twitter_data.iterrows():
    print(i)
    tokens = nltk.word_tokenize(row[2])
    tagged = nltk.pos_tag(tokens)
    for tag in tagged:
        word = tag[0].lower()
        if tag[1] in NOUNS and word in HUMAN_WORDS and word not in non_gendered\
            and word not in gendered and is_word(word):
            is_gendered = False
            for pi in Sp:
                if graph_has_is_a(word, pi):
                    is_gendered = True
                    break
            if is_gendered:
                gendered.add(word)
                print('GENDERED!!!!', word)
                gendered_file.write(word + '\n')
            else:
                non_gendered.add(word)
    gendered_file.flush()
gendered_file.close()

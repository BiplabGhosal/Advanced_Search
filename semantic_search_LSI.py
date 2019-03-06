from gensim import corpora,models, similarities
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from nltk.corpus import stopwords
import logging
import string
import nltk
import pandas as pd
import re


#Increasing the column width of individual columns in dataframes
pd.options.display.max_colwidth = 500

#Reading the hotel-reviews dataset in pandas dataframe
data = pd.read_csv(r'C:/Users/Billu/Downloads/hotel-reviews/data.csv', engine = 'python')

#Creating a list of unique stop words from NLTK's stopwords repository
stop = set(stopwords.words('english'))

#Punctuation marks were also added to use for removal in the corpus if needed (Not used currently)
stopwords_punctuation = stop.union(string.punctuation)


#Function to convert the input docs to lower case and also to remove all the special characters except 'period',
#as it is needed for sentence tokenization to create input lists for the model

def cleanData(sentence):
    processedList = ""
    # convert to lowercase, ignore all special characters - keep only alpha-numericals and spaces (not removing full-stop here)
    sentence = re.sub(r'[^A-Za-z0-9\s.]',r'',str(sentence).lower())
    sentence = re.sub(r'\n',r' ',sentence)

    # remove stop words
    sentence = " ".join([word for word in sentence.split() if word not in stop])
    return sentence
	
# Applying the function to all the values in "Description" column of dataframe

data['Description_New'] = data['Description'].map(lambda x: cleanData(x))

#Creation of lists based on each sentence (sentence tokenization)
tmp_corpus = data['Description_New'].map(lambda x: x.split('.'))

pre_corpus = []

# Creation of list of lists of unique words for each sentence, this will be used as input for the model
for i in range(len(tmp_corpus)):
    for line in tmp_corpus[i]:
        words = [x for x in line.split()]
        pre_corpus.append(words)
		
# Creating phrases from the input corpus to identify bigrams		
phrases = Phrases(sentences=pre_corpus,min_count=25,threshold=50)

# Creating the bigrams based on the extracted phrases
bigram = Phraser(phrases)

# Applying bi-gram conversion on the sentences
for index,sentence in enumerate(pre_corpus):
    pre_corpus[index] = bigram[sentence]

# Creating a dictionary from the words on pre-processed corpus
dictionary = corpora.Dictionary(pre_corpus)
print(dictionary.token2id) # prints unique words with their ID's

# Converting the sentences in corpus to Sparse vectors for feeding the LSI model,
# Bag of words technique is used for vectorization
corpus = [dictionary.doc2bow(text) for text in pre_corpus]

# Creation of LSI vector space based on the corpus and dictionary
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=50)

# Mapping the words in corpus to LSI space
index = similarities.MatrixSimilarity(lsi[corpus])

# Search function to create a vectorial representation of user supplied query and to capture
# the top 5 results of retrieved documents

def search(a):
	new_doc = a
	new_vec = dictionary.doc2bow(new_doc.lower().split())
	print(new_vec)
	vec_lsi = lsi[new_vec]
	print(vec_lsi)
	sims = index[vec_lsi]
	# Sorting the results based on top probability scores
	sims = sorted(enumerate(sims), key=lambda item: -item[1])	
	for i,j in enumerate(sims):
		if i<=5:
			print(pre_corpus[j[0]])
			#Joining the list of words to sentences from the results
			join_word = " ".join([word for word in pre_corpus[j[0]]])
			print(join_word)
			#Retrieval of results from actual dataframe column "Description"
			print(data.loc[data.Description_New.str.contains(join_word),'Description'])
			print('\n\n\n')
		else:
			break;
			
search('which is the best hotel')

import gensim
from gensim import corpora, models
from nltk.corpus import stopwords


def lda_topics(self, num_topics=5):
    """Get the topics in the text using LDA"""
    # Prepare the text
    texts = [self.tokens]

    # Create a corpus from a list of texts
    common_dictionary = corpora.Dictionary(texts)
    common_corpus = [common_dictionary.doc2bow(text) for text in texts]

    # Train the model
    lda = models.LdaModel(
        common_corpus, num_topics=num_topics, id2word=common_dictionary)

    # Get the topics
    topics = lda.print_topics(num_words=5)

    return topics



from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')

stemmer = PorterStemmer()
text = "Elephants are truly magnificent creatures"
for word in text.split(" "):
    print(f"{word}\t -> \t{stemmer.stem(word)}")



# Example from https://polyglot.readthedocs.io/en/latest/MorphologicalAnalysis.html
!polyglot download morph2.en morph2.ar

from polyglot.downloader import downloader
from polyglot.text import Text, Word

words = ["preprocessing", "processor", "invaluable", "thankful", "crossed"]
for w in words:
  w = Word(w, language="en")
  print("{:<20}{}".format(w, w.morphemes))


from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
text = "Elephants are truly magnificent creatures"
for word in text.split(" "):
    print(f"{word}\t -> \t{lemmatizer.lemmatize(word)}")


import nltk
from nltk import bigrams
from collections import Counter

nltk.download('punkt')

words = nltk.word_tokenize(text)  # 'text' is your document
bi_grams = list(bigrams(words))

bigram_freq = Counter(bi_grams)

print(bigram_freq.most_common(10))  # prints the 10 most common bigrams

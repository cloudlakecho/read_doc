
# topic.py - Latent Semantic Indexing Model using Truncated SVD
#

# To do
#
# Error
#   Some title are only numbers
#   Progress bar implementation
#

# Reference:
#   https://www.kaggle.com/thebrownviking20/topic-modelling-with-spacy-and-scikit-learn

# Usual imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import pdb, string
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import concurrent.futures
import time
import pyLDAvis.sklearn
from pylab import bone, pcolor, colorbar, plot, show, rcParams, savefig
import warnings
warnings.filterwarnings('ignore')

import os


# spaCy based imports
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

EARLY_DEBUGGING = False
DEBUGGING = True
EARLY_TESTING = False
TESTING = True


# Intput:
#   Clinical features of culture-proven Mycoplasma pneumoniae infections at King Abdulaziz University Hospital, Jeddah, Saudi Arabia
# Output:
#   clinical features culture proven mycoplasma pneumoniae infections king abdulaziz university hospital jeddah saudi arabia
def spacy_tokenizer(parser, sentence):
    try:
        mytokens = parser(sentence)
    except Exception as e:
        # print (e.args)
        # print ("Sentence: %s".format(sentence))
        # count_empty_title += 1
        return None

    try:
        mytokens = [ word.lemma_.lower().strip()
            if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    except Exception as e:
        print (e.args)
        pdb.set_trace()

    try:
        mytokens = [ word for word in mytokens
            if word not in stopwords and word not in punctuations ]
    except Exception as e:
        print (e.args)
        pdb.set_trace()
    mytokens = " ".join([i for i in mytokens])

    if (EARLY_DEBUGGING):
        pdb.set_trace()

    return mytokens


def summarize_doc(input):
    wines = pd.read_csv(input)
    count_empty_title = 0

    # Loading data

    print (wines.head())
    print (list(wines))

    # Creating a spaCy object
    nlp = spacy.load('en_core_web_lg')

    if (EARLY_DEBUGGING):
        pdb.set_trace()

    if (EARLY_DEBUGGING):
        pdb.set_trace()

    punctuations = string.punctuation
    stopwords = list(STOP_WORDS)

    if (TESTING):
        doc = nlp(wines['title'][3])
        review = str(" ".join([i.lemma_ for i in doc]))

        # Why?
        doc = nlp(review)
        if (EARLY_TESTING):
            print (doc)

        # spacy.displacy.render(doc, style='ent')

        # POS taggingwine
        # check NLP tool
        # PROPN: Noun of proposition
        # ADP: ?
        for i in nlp(review):
            print(i,"=>",i.pos_)

    # Parser for reviews
    #   clinical features culture proven mycoplasma pneumoniae infections king abdulaziz university hospital jeddah saudi arabia
    parser = English()


    tqdm.pandas()

    # Error
    # TypeError: object of type 'float' has no len()
    # find another way to show progress
    wines["unknown_one"] = wines["title"].progress_apply(spacy_tokenizer)

    # Creating a vectorizer
    vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english',
        lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')

    data_vectorized = vectorizer.fit_transform(wines["abstract"])

    # Why?
    NUM_TOPICS = 10

    # Latent Dirichlet Allocation Model
    lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10,
        learning_method='online',verbose=True)
    data_lda = lda.fit_transform(data_vectorized)

    # Non-Negative Matrix Factorization Model
    nmf = NMF(n_components=NUM_TOPICS)
    data_nmf = nmf.fit_transform(data_vectorized)

    # Latent Semantic Indexing Model using Truncated SVD
    lsi = TruncatedSVD(n_components=NUM_TOPICS)
    data_lsi = lsi.fit_transform(data_vectorized)

    # Functions for printing keywords for each topic
    def selected_topics(model, vectorizer, top_n=10):
        for idx, topic in enumerate(model.components_):
            print("Topic %d:" % (idx))
            print([(vectorizer.get_feature_names()[i], topic[i])
                            for i in topic.argsort()[:-top_n - 1:-1]])

    # Keywords for topics clustered by Latent Dirichlet Allocation
    print("LDA Model:")
    selected_topics(lda, vectorizer)

    # Keywords for topics clustered by Latent Semantic Indexing
    print("NMF Model:")
    selected_topics(nmf, vectorizer)

    # Keywords for topics clustered by Non-Negative Matrix Factorization
    print("LSI Model:")
    selected_topics(lsi, vectorizer)

    # Transforming an individual sentence
    text = spacy_tokenizer("Aromas include tropical fruit, \
        broom, brimstone and dried herb. The palate isn't \
        overly expressive, offering unripened apple, citrus and dried sage alongside brisk acidity.")
    x = lda.transform(vectorizer.transform([text]))[0]
    print(x)


    # Visualizing LDA results with pyLDAvis

    pyLDAvis.enable_notebook()
    dash = pyLDAvis.sklearn.prepare(lda, data_vectorized, vectorizer, mds='tsne')
    dash


    # Visualizing LSI(SVD) scatterplot
    svd_2d = TruncatedSVD(n_components=2)
    data_2d = svd_2d.fit_transform(data_vectorized)

    trace = go.Scattergl(
        x = data_2d[:,0],
        y = data_2d[:,1],
        mode = 'markers',
        marker = dict(
            color = '#FFBAD2',
            line = dict(width = 1)
        ),
        text = vectorizer.get_feature_names(),
        hovertext = vectorizer.get_feature_names(),
        hoverinfo = 'text'
    )
    data = [trace]
    iplot(data, filename='scatter-mode')

    ## The text version of scatter plot looks messy but you can zoom it for great results

    trace = go.Scattergl(
        x = data_2d[:,0],
        y = data_2d[:,1],
        mode = 'text',
        marker = dict(
            color = '#FFBAD2',
            line = dict(width = 1)
        ),
        text = vectorizer.get_feature_names()
    )
    data = [trace]
    iplot(data, filename='text-scatter-mode')

    # ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
    count_empty_title = 0
    def spacy_bigram_tokenizer(phrase):
        try:
            doc = parser(phrase) # create spacy object
        except Exception as e:
            print (e.args)
            print ("Sentence: %s".format(sentence))
            # count_empty_title += 1
            return None

        token_not_noun = []
        notnoun_noun_list = []
        noun = ""

        for item in doc:
            if item.pos_ != "NOUN": # separate nouns and not nouns
                token_not_noun.append(item.text)
            if item.pos_ == "NOUN":
                noun = item.text

            for notnoun in token_not_noun:
                notnoun_noun_list.append(notnoun + " " + noun)

        return " ".join([i for i in notnoun_noun_list])

    tqdm.pandas()

    # Error
    # TypeError: object of type 'float' has no len()
    wines["unknown_second"] = wines["title"].progress_apply(spacy_bigram_tokenizer)

    bivectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english',
        lowercase=True, ngram_range=(1,2))
    bigram_vectorized = bivectorizer.fit_transform(wines["abstract"])

    ## LDA for bigram data

    bi_lda = LatentDirichletAllocation(n_components=NUM_TOPICS,
        max_iter=10, learning_method='online',verbose=True)
    data_bi_lda = bi_lda.fit_transform(bigram_vectorized)

    ### Topics for bigram model

    print("Bi-LDA Model:")
    selected_topics(bi_lda, bivectorizer)

    bi_dash = pyLDAvis.sklearn.prepare(bi_lda, bigram_vectorized,
        bivectorizer, mds='tsne')
    bi_dash


def main():
    given_dir = "/home/cloud/data/covid_19"
    print(os.listdir(given_dir))
    wines = os.path.join(given_dir, 'metadata.csv')
    summarize_doc(wines)


if __name__ == '__main__':
    main()

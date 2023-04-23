
# topic.py - Latent Semantic Indexing Model using Truncated SVD
#     Singular Value Decomposition
#     Anxp= Unxn Snxp VTpxp
        #
        # Where
        # UTU = Inxn
        # VTV = Ipxp  (i.e. U and V are orthogonal)
        #
        # Where the columns of U are the left singular vectors
        # (gene coefficient vectors)
        # S (the same dimensions as A) has singular values and is diagonal
        # (mode amplitudes)
        # and VT has rows that are the right singular vectors
        # (expression level vectors).
        # The SVD represents an expansion of the original data in
        # a coordinate system where the covariance matrix is diagonal.

# To do
#   CountVectorizer vs. nlp
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
import sklearn
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

EARLY_DEBUGGING = True
DEBUGGING = True
EARLY_TESTING = False
TESTING = True


# Intput:
#   Clinical features of culture-proven Mycoplasma pneumoniae infections at King Abdulaziz University Hospital, Jeddah, Saudi Arabia
# Output:
#   clinical features culture proven mycoplasma pneumoniae infections king abdulaziz university hospital jeddah saudi arabia
def spacy_tokenizer(parser, sentence, stopwords, punctuations):
    try:
        mytokens = parser(sentence)
    except Exception as e:
        # print (e.args)
        # print ("Sentence: %s".format(sentence))
        # count_empty_title += 1
        # pdb.set_trace()
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

    return mytokens


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


# +++++ +++++ +++++ +++++ +++++
# (3) Make summary of paper
# paper title | total page | summary
#
# Comparing papers
# Latent Semantic Indexing Model using Truncated SVD -> Longest common
#   subsequence problem or Longest common substring problem
def summarize_doc(input, par):
    wines = pd.read_csv(input)
    wines['unknown_one'] = ""
    wines['unknown_second'] = ""
    count_empty_title = 0

    # Loading data
    print (wines.head())
    print (list(wines))

    # Creating a spaCy object
    # Required to run $ python -m spacy download en_core_web_sm
    nlp = spacy.load('en_core_web_lg')

    punctuations = string.punctuation
    stopwords = list(STOP_WORDS)

    if (TESTING):
        doc = nlp(wines['title'][3])  # word2vec: doc.vector will be (300, )
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

    # One row by one row
    for i_dx, i in tqdm(enumerate(wines['title']), desc='Reading meta data'):
        wines["unknown_one"][i_dx] = spacy_tokenizer(parser, i, stopwords
        , punctuations)

    # nan_rows = wines[wines.isnull().T.any().T]
    #
    # Looks like every row has NaN value, so skip this
    #
    # wines = wines.drop(nan_rows.index)

    # Creating a vectorizer
    #
    # To do
    #    Find difference from nlp (word2vec) function
    #
    vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english',
        lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
    nan_rows = wines[wines['abstract'].isnull()]
    wines = wines.drop(nan_rows.index)
    # <class 'scipy.sparse._csr.csr_matrix'>
    # (Pdb) data_vectorized
    # <107032x48341 sparse matrix of type '<class 'numpy.int64'>'
	# with 8340068 stored elements in Compressed Sparse Row format>
    data_vectorized = vectorizer.fit_transform(wines["abstract"])

    # Why specific number?
    NUM_TOPICS = par['no of topic']

    # Latent Dirichlet Allocation Model
    # What is proper number of iteration?
    lda = LatentDirichletAllocation(n_components=NUM_TOPICS,
        max_iter=par['iteration'],
        learning_method='online',verbose=True)
    # input total row by number of topic like 107032 by 10
    data_lda = lda.fit_transform(data_vectorized)

    # Non-Negative Matrix Factorization Model
    nmf = NMF(n_components=NUM_TOPICS)
    data_nmf = nmf.fit_transform(data_vectorized)

    # Latent Semantic Indexing Model using Truncated SVD
    lsi = TruncatedSVD(n_components=NUM_TOPICS)
    data_lsi = lsi.fit_transform(data_vectorized)

    if (EARLY_DEBUGGING):
        pdb.set_trace()

    # Functions for printing keywords for each topic
    def selected_topics(model, vectorizer, top_n=10):
        for idx, topic in enumerate(model.components_):
            print("Topic %d:" % (idx))
            #
            # Error spot
            # AttributeError: 'CountVectorizer' object has no attribute 'get_feature_names'
            #
            print([(vectorizer.get_feature_names()[i], topic[i])
                            for i in topic.argsort()[:-top_n - 1:-1]])

    # Keywords for topics clustered by Latent Dirichlet Allocation
    print("LDA Model:")
    #
    # Error spot
    # AttributeError: 'CountVectorizer' object has no attribute 'get_feature_names'
    #
    selected_topics(lda, vectorizer)

    # Keywords for topics clustered by Latent Semantic Indexing
    print("NMF Model:")
    selected_topics(nmf, vectorizer)

    # Keywords for topics clustered by Non-Negative Matrix Factorization
    print("LSI Model:")
    selected_topics(lsi, vectorizer)

    # +++++ +++++
    # Jupyter Notebook?

    pdb.set_trace()

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

    tqdm.pandas()

    # Error
    # TypeError: object of type 'float' has no len()
    for i_dx, i in tqdm(enumerate(wines['title'])):
        wines["unknown_two"][i_dx] = spacy_bigram_tokenizer(parser, i,
            stopwords, punctuations)

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


# ----- ----- ----- -----
# Transforming an individual sentence
def unit_test(par):
    # Why specific number?
    NUM_TOPICS = par['no of topic']

    # Latent Dirichlet Allocation Model
    # What is proper number of iteration?
    lda = LatentDirichletAllocation(n_components=NUM_TOPICS,
        max_iter=par['iteration'],
        learning_method='online',verbose=True)

    #
    # Error spot
    # TypeError: spacy_tokenizer() missing 3 required positional arguments: 'sentence', 'stopwords', and 'punctuations'
    #
    text = spacy_tokenizer("Aromas include tropical fruit, \
        broom, brimstone and dried herb. The palate isn't \
        overly expressive, offering unripened apple, citrus and dried sage alongside brisk acidity.")
    x = lda.transform(vectorizer.transform([text]))[0]
    print(x)

    if (EARLY_TESTING):
        pdb.set_trace()

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
def main():
    par = {'iteration': 2, 'no of topic': 10}

    unit_test(par)

    given_dir = "/home/cloud/data/covid_19"
    print(os.listdir(given_dir))
    wines = os.path.join(given_dir, 'metadata.csv')
    summarize_doc(wines, par)


if __name__ == '__main__':
    main()

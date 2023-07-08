
# main.py - Make group of research papers: clustering
#   (1) summerize each research papers (topic function)
#   (2)
#   (3)

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# COVID-19 Open Research Dataset Challenge (CORD-19)
# What has been published about information sharing and inter-sectoral collaboration?
#
# Cloud Cho in May 5, 2020
#
# How to run this code:
#
# Work?
#
# Error
#   some in "topic.py"
#
# To do
#    Groupping the document from
#
# Runtime environment
#    read_doc_py_3_8 Anaconda
#
#
# Reference:
#   Task: https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=583

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os, pdb, sys
# pyldavis required Python 3.8 for Anaconda
import topic



EARLY_DEBUGGING = False
DEBUGGING = True
EARLY_TESTING = False
TESTING = True

# if summary already done (basically topic function)
#   This file will exist
# FILE = "/home/cloud/data/covid_19/metadata.csv"
FILE = "/home/cloud/computer_programming/python/china_virus/read_doc/result/hrdata.csv"

# (1) Start chatbot
# She will receive a question and digest it to process

# (2) Input of each paper

# Latent Semantic Indexing Model using Truncated SVD
if (not os.path.exists(FILE)):
    par = {'iteration': 2, 'no of topic': 10}
    given_dir = "/home/cloud/data/covid_19"
    wines = os.path.join(given_dir, 'metadata.csv')
    topic.summarize_doc(wines, par)  # calling topic.py file

# To do
# where is the folder?
if (os.path.exists('/kaggle/input')):
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            # print(os.path.join(dirname, filename))
            if (DEBUGGING):
                pdb.set_trace()


# (3) Make summary of paper
# paper title | total page | summary
df = pd.read_csv(FILE)
pdb.set_trace()

#
# Comparing papers -> put in a table
# Method 1
#   Latent Semantic Indexing Model using Truncated SVD --> Longest common
#   subsequence problem or Longest common substring problem
#
# Method 2
#   Keyword matching --> Latent Semantic Indexing Model using Truncated SVD -->
#   Longest common subsequence problem or Longest common substring problem
#
# papser name | similar | not similar
#   similarity by keyword matching count?

# (4) Receive question
# Make list of synomym of key word
#   How chatbot understand question?


# (5) Find document using summary
# Check paper title
# Check summary
#   Yes: read document
#   No:


# (6) Read document to answer the question
#   How chatbot answer?

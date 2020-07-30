
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# COVID-19 Open Research Dataset Challenge (CORD-19)
# What has been published about information sharing and inter-sectoral collaboration?

# Cloud Cho in May 5, 2020

# Reference:
#   Task: https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=583

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# (1) Start chatbot
# She will receive a question and digest it to process

# (2) Input of each paper

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        # print(os.path.join(dirname, filename))


# (3) Make summary of paper
# paper title | total page | summary
#
# Comparing papers
# Latent Semantic Indexing Model using Truncated SVD -> Longest common
#   subsequence problem or Longest common substring problem


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

import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from main import review_to_words
# nltk.download() # download text data sets, including stop words
from nltk.corpus import stopwords # get stopwords list
# print(stopwords.words("english"))

train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

# print(train.shape) # should reutrn (25000, 3)
# print(train.columns.values) # should return array([id, sentiment, review], dtype=object)
# print(train["review"][0])
'''
example1 = BeautifulSoup(train["review"][0])
# print(train["review"][0])
# print(example1.get_text())

letters_only = re.sub("[^a-zA-Z]", " ", example1.get_text())
# print(letters_only)

lower_case = letters_only.lower()
words = lower_case.split()

words = [w for w in words if not w in stopwords.words("english")]
# print(words)
'''
num_reviews = train["review"].size
clean_train_reviews = []
# loop through each review

for i in range(0, num_reviews):
    # call function for each one, and add result to list
    clean_train_reviews.append(review_to_words(train["review"][i]))

print("Creating the bag of words...\n")
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the CountVectorizer" object, which is scikit-learn's bag of words tool
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()

# check out words in the vocab
vocab = vectorizer.get_feature_names()

print("Training the random forest...")
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)

# Fit the forest to the training set, using the bag of words as features and the sentiment labels as the response variable
# This may take a few minutes to run
forest = forest.fit(train_data_features, train["sentiment"])

import numpy as np # Make sure that numpy is imported
import pandas as pd
from part2 import review_to_wordlist

def makeFeatureVec(words, model, num_features):
    # Function to average all of the owrd vectors in a given paragraph
    # Pre-init an empty numpy arary (for speed) 
    featureVec = np.zeros((num_features, ), dtype="float32")
    nwords = 0.
    # Index2word is a list that contains the names of the words in the model's vocab. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    # Loop over each word in the review and, if it is in the model's vocab, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvfFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate the average feature vector for each one and return a 2D numpy array
    # Initialize a counter
    counter = 0.
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    # Loop through the reviews
    for review in reviews:
        # Print a status message every 1000th review
        if counter % 1000. == 0.:
            print("Review %d of %d" % (counter, len(reviews)))
        # Call the function defined above that makes average feature vectors
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        # Increment the counter
        counter = counter + 1
    return reviewFeatureVecs

train = pd.read_csv( "data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3, encoding="utf8" )
test = pd.read_csv( "data/testData.tsv", header=0, delimiter="\t", quoting=3, encoding="utf8"  )
unlabeled_train = pd.read_csv( "data/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3, encoding="utf8"  )

# Verify the number of reviews that were read (100,000 in total)
# print("Read %d labeled train reviews, %d labeled test reviews, " \
# "and %d unlabeled reviews\n" % (train["review"].size,  
# test["review"].size, unlabeled_train["review"].size ))
sentences = []  # Initialize an empty list of sentences

print("Parsing sentences from training set")
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)


# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

from gensim.models import word2vec
print("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)

'''
Calculate average feature vectors for training and testing sets, using the functions we defined above. Notice that we now use stop word removal
'''
clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))

trainDataVecs = getAvfFeatureVecs(clean_train_reviews, model, num_features)

print("Creating average feature vecs for test reviews")
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True))

testDataVecs = getAvgGeatureVecs(clean_test_reviews, model, num_features)

# Fit a random forest to the training data, using 100 trees
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)
print("Fitting a random forest to labeled training data...")
forest = forest.fit(trainDataVecs, train["sentiment"])

# Test & extract results
result = forest.predict(testDataVecs)

# Write the test results
output = pd.DataFrame(data={"id":test["id"], "sentiment": result})
output.to_csv("Word2Vec_AverageVectors.csv", index=False, quoting=3)

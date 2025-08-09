# Load libraries
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Create text
text_data = np.array([
    'I love Brazil. Brazil!',
    'Brazil is best',
    'Germany beats both'
])
print("text_data:", text_data)

# Create bag of words
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)
print("Vocabulary:", count.vocabulary_)
print("bag_of_words (sparse matrix):\n", bag_of_words)

# Create feature matrix
features = bag_of_words.toarray()
print("features (array):\n", features)

# Create target vector (labels)
target = np.array([0, 0, 1])
print("target:", target)

# Create Naive Bayes classifier with priors
classifier = MultinomialNB(class_prior=[0.25, 0.5])

# Train model
model = classifier.fit(features, target)

# ---- Predict new sentences ----
new_text = np.array([
    "Brazil is amazing",
    "Germany is the winner",
    "I love football"
])
print("new_text:", new_text)

# Convert new text to bag of words (use the same vectorizer!)
new_features = count.transform(new_text)
print("new_features (array):\n", new_features.toarray())

# Predict classes
predictions = model.predict(new_features)
print("predictions:", predictions)

# Show results
for sentence, label in zip(new_text, predictions):
    print(f"'{sentence}' → Predicted class: {label}")

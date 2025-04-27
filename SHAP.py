#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 14:48:20 2025

@author: anyakondamani
"""
#%% IMPORTS 
from __future__ import print_function
import shap
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.ensemble
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import fetch_20newsgroups
from collections import defaultdict
from nltk.tokenize import word_tokenize
from scipy.signal import find_peaks
import nltk
nltk.download('punkt')
#%% Importing Data & Training SGDClassifier
categories = ['alt.atheism', 'soc.religion.christian'] # Categories of interest.

newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

class_names = ['atheism', 'christian'] # Setting outcome class names. 

# Initializing & fitting tf-idf vectorizer.
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=None)
X_train_tfidf = tfidf_vectorizer.fit_transform(newsgroups_train.data)
X_test_tfidf = tfidf_vectorizer.transform(newsgroups_test.data)
y_train = newsgroups_train.target
y_test = newsgroups_test.target

# Training & fitting the classifier. 
sgd_clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train_tfidf, y_train)

# Confusion Matrix
y_pred = sgd_clf.predict(X_test_tfidf)
conf_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
print(conf_matrix)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for SGDClassifier")
plt.show()

#%% Using SHAP's Explainer
# Initializing the explainer & applying it to the test data.
explainer = shap.LinearExplainer(sgd_clf, X_test_tfidf, feature_perturbation="interventional")
shap_values = explainer(X_test_tfidf)
 
# Selecting 5 documents from the test set to explain.
indices = [0, 85, 110, 279, 377] # For learning purposes, 5 chosen documents include some false predictions.
for i in indices:
    print(f"Document {i} - True Label: {class_names[y_test[i]]}, Predicted: {class_names[y_pred[i]]}")
for i in indices:
    shap.force_plot(explainer.expected_value, shap_values[i].values, tfidf_vectorizer.get_feature_names_out(), matplotlib=True, show=False, contribution_threshold=0.02)
    plt.title(f"SHAP Force Plot for Document {i}", loc='left')
    plt.show()

# Accuracy of the classifier and overall number of misclassified documents.
misclassified_indices = np.where(y_pred != y_test)[0]
n_misclassified = len(misclassified_indices)
print(f"Number of misclassified documents: {n_misclassified}")
accuracy = sgd_clf.score(X_test_tfidf, y_test)
print(f"SGD Classifier Accuracy: {accuracy:.4f}")

# conf_i - the difference between the probabilities of the two predicted classes for a document.
# Calculating conf_i for all misclassified documents and visualizing the distribution.
probs = sgd_clf.decision_function(X_test_tfidf)
conf_i = np.abs(probs[misclassified_indices])
plt.figure(figsize=(8, 5))
sns.histplot(conf_i, bins=30, kde=True)
plt.xlabel("Confidence Difference (|conf_i|)")
plt.ylabel("Frequency")
plt.title("Distribution of Confidence Differences for Misclassified Documents")
plt.show()


# For each word (word_j), the number of documents it helped misclassify is called is count_j.
# The total weight of that word in all documents it helped misclassify is called weight_j.

# Identifying tokens (word_j) that contributed to the misclassification of documents.
counts = defaultdict(int)

for idx in misclassified_indices:

  vals = shap_values[idx].values
  feats = tfidf_vectorizer.get_feature_names_out()
  predicted_label = class_names[y_pred[idx]]

  for feat_i, val_i in zip(feats, vals):
    if predicted_label == 'christian' and val_i > 0:
      counts[feat_i] += 1
    elif predicted_label == 'atheism' and val_i < 0:
      counts[feat_i] += 1

word_j = list(counts.items())

weights = defaultdict(float)

for idx in misclassified_indices:

  vals = shap_values[idx].values
  feats = tfidf_vectorizer.get_feature_names_out()
  predicted_label = class_names[y_pred[idx]]

  for feat_i, val_i in zip(feats, vals):
    if predicted_label == 'christian' and val_i > 0:  # words wrongly contributing to christian (1) classification
      weights[feat_i] += abs(val_i)
    elif predicted_label == 'atheism' and val_i < 0:  # words wrongly contributing to atheist (0) classification
      weights[feat_i] += abs(val_i)
      
      
# Computing the number of documents (count_j) these words helped to misclassify.
count_j = list(counts.values())
print(len(word_j))

# Computing the sum of absolute SHAP values (weight_j) for each word_j.
weight_j = list(weights.values())

# Distribution of count_j. 
plt.figure(figsize=(10, 5))
ax_count = sns.histplot(counts, bins=30, kde=True)
plt.xlabel("Number of Documents (count_j)")
plt.ylabel("Frequency")
plt.title("Distribution of count_j (Word Contribution Frequency)")
plt.show()

# Distribution of weight_j.
plt.figure(figsize=(10, 5))
ax_weight = sns.histplot(weights, bins=30, kde=True)
plt.xlabel("Total SHAP Weight (weight_j)")
plt.ylabel("Frequency")
plt.title("Distribution of weight_j (Total Contribution Weight)")
plt.show()

#%% Implementing Feature Selection

# Determining Peaks in count_j.
kde_line = ax_count.lines[0]
x_kde = kde_line.get_xdata()
y_kde = kde_line.get_ydata()
peaks, _ = find_peaks(y_kde, prominence=0.1)

plt.figure(figsize=(10, 5))
sns.histplot(counts, bins=30, kde=True, alpha=0.3)

plt.scatter(x_kde[peaks], y_kde[peaks], color='red', label='Peaks')

for i, (x, y) in enumerate(zip(x_kde[peaks], y_kde[peaks])):
    plt.text(x, y, f'Peak {i+1}\n({x:.2f}, {y:.2f})',
             ha='center', va='bottom', color='red',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.legend()
plt.xlabel("Number of Documents (count_j)")
plt.ylabel("Frequency")
plt.title("Detecting Peaks")
plt.show()

# Determining Drop-off in weight_j.
kde_line = ax_weight.lines[0]
x_kde = kde_line.get_xdata()
y_kde = kde_line.get_ydata()

slope = np.gradient(y_kde, x_kde)

mask = (x_kde >= 0.01) & (x_kde <= 0.3)
x_segment = x_kde[mask]
y_segment = y_kde[mask]
slope_segment = slope[mask]

drop_idx = np.argmin(slope_segment)
drop_x = x_segment[drop_idx]
drop_y = y_segment[drop_idx]

plt.figure(figsize=(10, 5))
sns.histplot(weights, bins=30, kde=True, alpha=0.4)

plt.scatter(drop_x, drop_y, color='darkred', s=100, label=f'Big Drop at x={drop_x:.2f}')
plt.axvline(drop_x, color='red', linestyle='--', alpha=0.6)

plt.text(drop_x, drop_y, f'Sharp Drop\nx={drop_x:.2f}',
         ha='right', va='top', color='darkred',
         bbox=dict(facecolor='white', alpha=0.8))

plt.xlabel("Number of Documents (weight_j)")
plt.ylabel("Frequency")
plt.title("Identifying the Largest Drop-off")
plt.legend()
plt.show()

# Implementing a strategy for feature selection using thresholds from peak & drop-off.
min_count_threshold = x_kde[0]
max_count_threshold = x_kde[1]
min_weight_threshold = drop_x
word_contributions = {
    word: {"count": counts[word], "weight": weights[word]}
    for word in counts
}
feature_names = tfidf_vectorizer.get_feature_names_out()

words_to_remove = [word for word, values in word_contributions.items() if (values["count"] < 3.88 or values["count"] > 40.35) and values["weight"] < 0.02]
print(f"Removing {len(words_to_remove)} words that contribute to misclassification.")

tfidf_vectorizer_filtered = TfidfVectorizer(stop_words='english', max_features=None, vocabulary=[w for w in feature_names if w not in words_to_remove])
X_train_tfidf_filtered = tfidf_vectorizer_filtered.fit_transform(newsgroups_train.data)
X_test_tfidf_filtered = tfidf_vectorizer_filtered.transform(newsgroups_test.data)

sgd_clf_filtered = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=1000, tol=1e-3)
sgd_clf_filtered.fit(X_train_tfidf_filtered, y_train)

accuracy_filtered = sgd_clf_filtered.score(X_test_tfidf_filtered, y_test)
print(f"SGD Classifier Accuracy after Feature Selection: {accuracy_filtered:.4f}")

# Obtaining predictions before and after feature selection.
y_pred_before = sgd_clf.predict(X_test_tfidf)
y_pred_after = sgd_clf_filtered.predict(X_test_tfidf_filtered)

improved_indices = np.where((y_pred_before != y_test) & (y_pred_after == y_test))[0]

print(f"Number of documents correctly classified after feature selection: {len(improved_indices)}")

# Observing a previously misclassified document.
if len(improved_indices) > 0:
    example_idx = improved_indices[0]
    original_text = newsgroups_test.data[example_idx]

    print(f"Document Index: {example_idx}")
    print(f"True Label: {y_test[example_idx]}")
    print(f"Predicted Before: {y_pred_before[example_idx]}")
    print(f"Predicted After: {y_pred_after[example_idx]}")
    print("\nDocument Text:\n")
    print(original_text[:1000])

X_example_before = X_test_tfidf[example_idx]
X_example_after = X_test_tfidf_filtered[example_idx]

explainer_before = shap.LinearExplainer(sgd_clf, X_test_tfidf, feature_perturbation="interventional")
explainer_after = shap.LinearExplainer(sgd_clf_filtered, X_test_tfidf_filtered, feature_perturbation="interventional")

shap_values_before = explainer_before(X_example_before)
shap_values_after = explainer_after(X_example_after)

feature_names_before = tfidf_vectorizer.get_feature_names_out()
feature_names_after = tfidf_vectorizer_filtered.get_feature_names_out()

print("\nSHAP Force Plot Before Feature Selection:")
plt.figure(0)
shap.force_plot(explainer_before.expected_value, shap_values_before.values, feature_names_before, matplotlib=True, show=False, contribution_threshold=0.02)
plt.title(f"SHAP Force Plot Before Feature Selection", loc='left')
plt.show()
# Force plot after feature selection.
print("\nSHAP Force Plot After Feature Selection:")
plt.figure(1)
shap.force_plot(explainer_after.expected_value, shap_values_after.values, feature_names_after, matplotlib=True, show=False, contribution_threshold=0.02)
plt.title(f"SHAP Force Plot After Feature Selection", loc='left')
plt.show()







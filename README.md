# Text Classification with SHAP
**Note:** This short project was completed as part of a homework submission for Responsible Data Science, Spring 2025 at New York University.

Dataset used: [20 newsgroups dataset](https://scikit-learn.org/stable/datasets/#the-20-newsgroups-text-dataset)

Source referenced for implementation: [SHAP Paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf)

### Implementing SHAP
SHAP's explainer is used to study the documents classified by the SGDClassifier. 
Initially, the classifier’s accuracy is 93.72%, misclassifying 45 documents from a total of 717. 
The value conf_i, denoting the difference between the probabilities of the two predicted classes, shows high frequencies near zero suggesting that many misclassifications happened with low confidence. A few larger values represent confident but wrong predictions. 

The distribution of count_j, representing the number of documents misclassified by each word contributing to misclassification, is bimodal. The left spike represents many words that are rare contributors to misclassification, whereas the right spike represents many words that appear often in misclassified documents. This implies two general types of words that might lead to misclassification: rare terms and generic ambiguous terms.

Similarly, the distribution of weight_j, representing the weight of each word in all the documents that contributed to misclassification, shows a long right tail. This suggests that most words carry low total influence, however, a few words have a large influence and likely contribute to multiple misclassifications with meaningful weight. 

### Feature Selection
The aim of feature selection is to improve the accuracy of the classifier using SHAP explanations.

The feature selection strategy used is designed to remove words that consistently contribute to classification errors across multiple documents. Specifically, it filters out words that occur too infrequently or frequently among misclassified documents (count_j) and have low total SHAP contribution across those documents (weight_j). These thresholds are determined using peak detection in the KDE plot of count_j and drop-off detection in the KDE of weight_j. By focusing on the values of count_j between peaks, the model retains words that are consistently but not overwhelmingly present in problematic cases. Even if a word shows up often, if its SHAP contribution is small, it's likely not meaningfully influencing the classification. 

After completing this feature selection and retraining the model, accuracy is improved from 93.72% to 94.00%. Additionally, 8 documents that were previously misclassified are now correctly classified. From these, Document 25 is selected to compare explanations before and after feature selection. This document’s true label is atheism, however, it was previously predicted to be christian. As seen in the before plot below, the words “aaron” and “york” are pushing the prediction toward christian. On the other side, words like “west” and “arabia” are pulling the prediction back toward atheism, but not strongly enough to counteract the effect of “aaron” and “york”. This imbalance caused the model to lean incorrectly toward the classification of christian. In the after plot, “aaron” has been removed. This means that the word “aaron” fell within the feature selection thresholds previously set. As a result, “west” and “arabia” are now more influential in pulling the prediction toward atheism, resulting in the correct classification.

BEFORE:
![image](https://github.com/user-attachments/assets/2d70462b-baa9-4292-9221-fd658d67cd41)
AFTER:
![image](https://github.com/user-attachments/assets/ee8f3995-ca30-478d-843e-74602ccfc578)


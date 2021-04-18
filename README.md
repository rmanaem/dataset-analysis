# Veracity Detection Using Supervised Learning

## Abstract

The dawn of the digital age and the World Wide Web paved the way for the dissemination of high volumes of unrefined information through online media outlets and social media platforms. Ease of access, corporate and political biases and lack of appropriate moderation in these mediums have resulted in the propagation of false information and fabricated news, forging an essential demand for filtration. This project aims at analyzing related data using supervised learning tools and techniques to develop models for veracity detection. This work explores different textual properties and patterns that appear in news articles to extract linguistic features for training models using classification algorithms.

---

## Introduction

The advent of technology has surged through all aspects of life revolutionizing the way we receive, process, and pass on information. A multitude of our decisions are based on the information reported and broadcasted through different news and media outlets. These outlets and other entities utilize social media platforms (Facebook, Twitter, …) to further disseminate information to a broader audience. In a world where news organizations and the press have their own specific corporate and political allegiances and our means of communication and sharing information are designed to reinforce pre-existing biases, it's essential to recognize the significance of validity of the available information. The 2016 US presidential election is widely used as a prime example to demonstrate the impact of disinformation on major events. Propagation of fake news through Facebook and Twitter in the months leading up to the election influenced voter turnout in democratic voters and voter decisions in apolitical voters altering the course of the election [1].

The task of validating news as essential as it may be can be time-consuming and difficult even for experts as it requires exhaustive research into different aspects. To tackle these challenges, automatic detection models have been researched by employing artificial intelligence and machine learning principles. One of the recent works proposed a novel hybrid deep learning model that combines convolutional and recurrent neural networks for fabricated news classification. This model was able to successfully validate two fake news datasets and achieve results that were significantly better than other non-hybrid baseline methods [2]. Another study utilized a machine learning ensemble approach to train a combination of different machine learning algorithms to expand their research beyond the domain of the existing literature. The ensemble learners showed promising results in a wide variety of applications as the learning models tended to reduce error rates using techniques such as bagging and boosting [3].

This project is aimed at automating the classification of news articles by training and testing machine learning models using supervised learning tools and techniques. This work explores keywords, phrases, and other textual properties that appear in news articles to extract linguistic features for training models using classification algorithms.

---

## Materials and Methods

For analysis, an imbalanced dataset with categorical features containing news articles from 2016 to 2017 was selected [4]. Dataset is comprised of two sets: a fake set containing 23503 fabricated articles (59.8 MB) and a true set containing 21418 authentic articles (51.8 MB). Attributes include title, text, subject, and the date of the article.

### Data Pre-processing

Data pre-processing was done utilizing tools from Pandas library [5], Natural Language Tool Kit (NLTK) [6], Spark [7], and Scikit-learn [8].
Using Pandas DataFrames, irrelevant features, subject, and date were removed and data points in sets were labeled separately. Each set was then randomly sampled for a specific goal.

| Sample Name | Number of True Data Points | Number of Fake Data Points | Goal |
| :-----: |     :-----:         |  :-----: | :-----:|
| Balanced_Sample1 | 3000 | 3000 | Building models and tuning parametes |
| Balanced_Sample2 | 15000 | 15000 | Large-scale analysis effect of data        imbalance |
| Balanced_Sample3 | 21000 | 21000 | Resolving data imbalance and final training and testing of models |
| Imbalanced_Sample1 | 2000 | 4000 | Small-scale analysis of effect of data imbalance |
| Imbalanced_Sample2 | 10000 | 20000 | Large-scale analysis of effect of data imbalance |
| Imbalanced_Sample3 | 20000 | 10000 | Large-scale analysis of effect of data imbalance |
| Imbalanced_Sample4 | 21324 | 22314 | Dataset without anamolies |

Using Spark RDDs, data points were filtered to remove instances with missing features and anomalies. Title and text fields of each article were then concatenated to format data points into tuples containing content (title and text) and label, simplifying and speeding up processing in later stages. The content of each article was transformed into a vector of words by applying Spark's Tokenizer [9]. Sequence of tokens extracted by examining articles, stop words from NLTK's stop words module, and punctuation marks from Python's string module were used to filter out the noise and trivial tokens from the word vectors to allow for more robust feature extraction by focusing on important words.

### Feature Extraction

To convert the tokenized and filtered articles to a set of features utilized by classification algorithms, the term frequency-inverse document frequency (TF-IDF) technique was employed. TF-IDF is a feature vectorization method widely used in text mining to reflect the importance of a term to a document in the corpus. Term frequency of ***t*** in document ***d*** denoted by ***TF(t, d)*** is the number of times that term t appears in document ***d***, while document frequency of term ***t*** in corpus ***D*** denoted by ***DF(t, D)*** is the number of documents that contain term ***t***. Using term frequency alone to measure the importance would lead to emphasizing terms that appear very often but carry little information about the document. To resolve this issue Inverse document frequency is used as a numerical measure of how much information a term provides.

<p alt="IDF" align="center"><a href="https://spark.apache.org/docs/latest/ml-features#tf-idf"><img src="https://github.com/rmanaem/veracity-detection/blob/master/figures/idf.png?raw=true"/></a></p>

<p alt="TFIDF" align="center"><a href="https://spark.apache.org/docs/latest/ml-features#tf-idf"><img src="https://github.com/rmanaem/veracity-detection/blob/master/figures/tfidf.png?raw=true"/></a></p>

Spark’s TF-IDF is implemented in two steps:

- HashingTF: Transformer that takes sets of terms and converts those into fixed-length feature vectors. It utilizes a hashing trick, mapping a raw feature into an index (term) by applying a hashing function. Then, term frequencies are calculated based on the mapped indices.
- IDF: Estimator that fits on a dataset and produces an IDFModel. The IDFModel takes feature vectors produced by HashingTF and scales each feature [10].

### Defining Training and Test Sets

Training and test sets were defined by k-fold cross validation implemented using Scikit-learn’s KFold. The data was split into 5 folds and models were trained and tested 5 times. In each iteration, 4 folds are used to train and 1 was used to test the models. The average performance was computed to reflect the overall result.

### Classifiers

For the binary classification of the articles into True and False classes, k-nearest neighbors (kNN) and random forest (RF) classifiers were chosen. kNN was implemented using Scikit-learn’s KNeighborsClassifier [11]. The number of nearest neighbors for each sample was determined by the square root of the input and incremented if the result was even. The random forest classifier was implemented using Scikit-learn’s RandomForestClassifier [12]. Random forest parameters were tuned, resulting in an ensemble of 45 trees with a max depth of 20.

---

## Results

Numerous tests were conducted to analyze and compare models based on performance, sensitivity to the number of extracted features, the effect of data imbalance, and the effect of stratification using f1 score as the measure.

The comparison of models based on performance alone was conducted using a stratified sample of 42000 data points with a balanced class distribution. Both classifiers were able to produce near-perfect f1 scores but, as exhibited by the table and the plot below, RF outperformed kNN by more than 2%.

<table align="center">
    <tr>
        <th>Classifier</th>
        <th>Average F1 Score</th>
    </tr>
    <tr>
        <td>KNN</td>
        <td>0.9779454954</td>
    </tr>
    <tr>
        <td>RF</td>
        <td>0.9998809608</td>
    </tr>
</table>

<p alt="IDF" align="center"><a href="https://github.com/rmanaem/veracity-detection/tree/master/figures/classifier_performance.png"><img src="https://github.com/rmanaem/veracity-detection/blob/master/figures/classifier_performance.png?raw=true"/></a></p>

To examine the sensitivity of classifiers to the number of extracted features, a sample of 42000 data points with balanced class distribution was chosen. The sample was not stratified to make sure the performance of models was not affected by stratification. The number of features was increased from 10 to 50 and the average f1 score of classifiers was recorded at each stage. As illustrated by the table and the plot below, kNN numbers show a significant drop of more than 14% in the average f1 score as the number of features was increased while RF numbers show a very slight improvement in the performance of the model.

<table align="center">
    <tr>
        <th>Features</th>
        <th>kNN Average F1 Score</th>
        <th>RF Average F1 Score</th>
    </tr>
    <tr>
        <td>10</td>
        <td>0.9784331</td>
        <td>0.9997995</td>
    </tr>
    <tr>
        <td>20</td>
        <td>0.9632550</td>
        <td>0.9999662</td>
    </tr>
    <tr>
        <td>30</td>
        <td>0.9505998</td>
        <td>0.9999336</td>
    </tr>
    <tr>
        <td>40</td>
        <td>0.9289815</td>
        <td>0.9998652</td>
    </tr>
    <tr>
        <td>50</td>
        <td>0.8350678</td>
        <td>0.9999325</td>
    </tr>
</table>

<p alt="IDF" align="center"><a href="https://github.com/rmanaem/veracity-detection/tree/master/figures/classifier_sensitivity_to_features.png"><img src="https://github.com/rmanaem/veracity-detection/blob/master/figures/classifier_sensitivity_to_features.png?raw=true"/></a></p>

To see how models performed in presence of data imbalance, two samples containing 30000 data points, one with balanced class distribution and the other with a 2 to 1 class distribution ratio favoring the fake class were selected. The f1 score of each classifier was recorded at each iteration of k-fold cross-validation using each sample. As demonstrated by the table and the plot below, the performance of both classifiers was reduced when the dataset was imbalanced however this reduction was much more significant in the case of kNN (~1%) than RF (~0.0000962%).

<table align="center">
    <tr>
        <th>Classifier</th>
        <th>Balanced</th>
        <th>Average F1 Score</th>
    </tr>
    <tr>
        <td>kNN</td>
        <td>Yes</td>
        <td>0.9783698</td>
    </tr>
    <tr>
        <td>kNN</td>
        <td>No</td>
        <td>0.9675079</td>
    </tr>
    <tr>
        <td>RF</td>
        <td>Yes</td>
        <td>0.9997995</td>
    </tr>
    <tr>
        <td>RF</td>
        <td>No</td>
        <td>0.9997033</td>
    </tr>
</table>

<p alt="IDF" align="center"><a href="https://github.com/rmanaem/veracity-detection/tree/master/figures/effect_of_data_imbalance.png"><img src="https://github.com/rmanaem/veracity-detection/blob/master/figures/effect_of_data_imbalance.png?raw=true"/></a></p>

Lastly, models were tested to see the effect of stratification. To do so 21000 data points from each class were sampled and kept separate. After pre-processing, each set was then partitioned using k-fold cross-validation. The folds from each class were merged with their respective counterpart from the opposite class only before they were fed into the models for training and testing. As shown by the table and the plot below, stratification showed a very slight improvement in the overall results.

<table align="center">
    <tr>
        <th>Classifier</th>
        <th>Stratified</th>
        <th>Average F1 Score</th>
    </tr>
    <tr>
        <td>kNN</td>
        <td>Yes</td>
        <td>0.9779454</td>
    </tr>
    <tr>
        <td>kNN</td>
        <td>No</td>
        <td>0.9779412</td>
    </tr>
    <tr>
        <td>RF</td>
        <td>Yes</td>
        <td>0.9998809</td>
    </tr>
    <tr>
        <td>RF</td>
        <td>No</td>
        <td>0.9998569</td>
    </tr>
</table>

<p alt="IDF" align="center"><a href="https://github.com/rmanaem/veracity-detection/tree/master/figures/effect_of_stratification.png"><img src="https://github.com/rmanaem/veracity-detection/blob/master/figures/effect_of_stratification.png?raw=true"/></a></p>

---

## Discussion

The models were able to achieve promising results and even a near-perfect f1 score in the case of random forest. While tools and techniques utilized during pre-processing and feature extraction like removal of stop words and unimportant tokens as well as TF-IDF could be credited for the high performance, it’s important to take into account the effect that the nature of the dataset i.e. how and from where the articles were collected, whether they were artificially produced or manually written, might have had an effect on the performance. This effect could be measured by cross-referencing and comparing the results of this dataset with an already validated dataset with an equivalent schema.

Regardless of the nature of the data, random forest performed better and proved to be more stable. kNN’s underperformance is mainly because it relies on the distance between data points for prediction. This distance can be affected when the number of features is modified. Moreover, in presence of data imbalance, the majority class will have more weight in the prediction as it occupies a larger portion of the data space, leading to a rise in the number of false negatives or false positives. Random forest on the other hand is an ensemble of trees where each tree splits the data based on the features and classifies data points independent of one another leading to overall consistent performance.

To build on this work and its findings, possible future work could improve the models’ performance by exploring other forms of feature extraction namely those that take word semantics into account (e.g., word embedding). They could benefit from having larger and more versatile datasets to ensure that the performance of the models is not overshadowed by the nature of the dataset and lastly, they could take advantage of algorithms that are more stable in dealing with high-dimensional data to achieve better results.

---

## References

[1] A. Bovet and H. A. Makse, “Influence of fake news in Twitter during the 2016 US presidential election,” Nature News, 02-Jan-2019. [Online]. Available: <https://www.nature.com/articles/s41467-018-07761-2>. [Accessed: 18-Feb-2021].

[2] J. A. Nasir, O. S. Khan, and I. Varlamis, “Fake news detection: A hybrid CNN-RNN based deep learning approach,” International Journal of Information Management Data Insights, 05-Jan-2021. [Online]. Available: <https://www.sciencedirect.com/science/article/pii/S2667096820300070>. [Accessed: 18-Feb-2021].

[3] I. Ahmad, M. Yousaf, S. Yousaf, and M. O. Ahmad, “Fake News Detection Using Machine Learning Ensemble Methods,” Complexity, 17-Oct-2020. [Online]. Available: <https://www.hindawi.com/journals/complexity/2020/8885861>. [Accessed: 18-Feb-2021].

[4] C. Bisaillon, “Fake and real news dataset,” 26-Mar-2020. [Online]. Available: <https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset>. [Accessed: 1-Apr-2021].

[5] “pandas documentation,” pandas documentation - pandas 1.2.4 documentation. [Online]. Available: <https://pandas.pydata.org/docs/>. [Accessed: 1-Apr-2021].

[6] “Natural Language Toolkit,” Natural Language Toolkit - NLTK 3.6 documentation. [Online]. Available: <https://www.nltk.org/>. [Accessed: 1-Apr-2021].

[7] “Spark Overview,” Overview - Spark 3.1.1 Documentation. [Online]. Available: <https://spark.apache.org/docs/latest/>. [Accessed: 1-Apr-2021].

[8] “learn,” scikit. [Online]. Available: <https://scikit-learn.org/stable/>. [Accessed: 1-Apr-2021].

[9] “Extracting, transforming and selecting features,” Extracting, transforming and selecting features - Spark 3.1.1 Documentation. [Online]. Available: <https://spark.apache.org/docs/latest/ml-features#tokenizer>. [Accessed: 1-Apr-2021].

[10] “Feature Extraction and Transformation - RDD-based API,” Feature Extraction and Transformation - RDD-based API - Spark 3.1.1 Documentation. [Online]. Available: <https://spark.apache.org/docs/latest/mllib-feature-extraction.html#tf-idf>. [Accessed: 17-Apr-2021].

[11] “sklearn.neighbors.KNeighborsClassifier¶,” scikit. [Online]. Available: <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>. [Accessed: 1-Apr-2021].

[12] “sklearn.ensemble.RandomForestClassifier¶,” scikit. [Online]. Available: <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>. [Accessed: 1-Apr-2021].

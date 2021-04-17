# Veracity Detection Using Supervised Learning

## Abstract

Dawn of the digital age and the World Wide Web paved the way for dissemination of high volumes of unrefined information through online media outlets and social media platforms. Ease of access, corporate and political biases, and lack of appropriate moderation in these mediums has resulted in propagation of false information and fabricated news, forging an essential demand for filtration. This project aims at analyzing related data using supervised learning tools and techniques to develop models for veracity detection. This work explores different textual properties and patterns that appear in news articles to extract linguistic features for training models using classification algorithms.

## Introduction

The advent of technology has surged through all aspect of life revolutionizing the way we receive, process, and pass on information. Multitude of our decisions are based on the information reported and broadcasted through different news and media outlets. These outlets and other entities utilize social media platforms (Facebook, Twitter, …) to further disseminate information to a broader audience. In a world where news organizations and the press have their own specific corporate and political allegiances and our means of communication and sharing information are designed to reinforce pre-existing biases, its essential to recognize the significance of validity of the available information. The 2016 US presidential election is widely used as the prime example to demonstrate the impact of disinformation on major events. Propagation of fake news through Facebook and Twitter in the months leading up to the election influenced voter turnout in democratic voters and voter decision in apolitical voters altering the course of the election [1].

The task of validating news as essential as it may be, can be time consuming and difficult even for experts as It requires exhaustive research into different aspects. To tackle these challenges, automatic detection models have been researched by employing artificial intelligence and machine learning principles. One of the recent works proposed a novel hybrid deep learning model that combines convolutional and recurrent neural networks for fabricated news classification. This model was able to successfully validate two fake news datasets and achieve results that were significantly better than other non-hybrid baseline methods [2]. Another study utilized machine learning ensemble approach to train a combination of different machine learning algorithms to expand their research beyond the domain of the existing literature. The ensemble learners showed promising results in a wide variety of applications as the learning models had the tendency to reduce error rate using techniques such as bagging and boosting [3].

This project is aimed at automating classification of news articles by training and testing machine learning models using supervised learning tools and techniques. This work explores key words, phrases, and other textual properties that appear in news articles to extract linguistic features for training models using classification algorithms.

## Materials and Methods

For analysis, an imbalanced dataset with catergorical features containing news articles from 2016 to 2017 was selected []. Dataset is comprised of two sets: a fake set containing 23503 fabricated articles (59.8 MB) and a true set containing 21418 authentic articles (51.8 MB). Attributes include title, text, subject, and the date of the article.

### Data Pre-processing

Data pre-procesing was done utilizing tools from Pandas library [], Natural Language Tool Kit (NLTK) [], Spark [], and Scikit-larn [].
Using Pandas DataFrames irrelevant features, subject and date, were removed and data points in sets were labeled separately. Each set was then randomly sampled for a specific goal.

| Sample Name | Number of True Data Points | Number of Fake Data Points | Goal |
| :-----: |     :-----:         |  :-----: | :-----:|
| Balanced_Sample1 | 3000 | 3000 | Building models and tuning parametes |
| Balanced_Sample2 | 15000 | 15000 | Large-scale analysis effect of data        imbalance |
| Balanced_Sample3 | 21000 | 21000 | Resolving data imbalance and final training and testing of models |
| Imbalanced_Sample1 | 2000 | 4000 | Small-scale analysis of effect of data imbalance |
| Imbalanced_Sample2 | 10000 | 20000 | Large-scale analysis of effect of data imbalance |
| Imbalanced_Sample3 | 20000 | 10000 | Large-scale analysis of effect of data imbalance |
| Imbalanced_Sample4 | 21324 | 22314 | Dataset without anamolies |

Using Spark RDDs data points were filtered to remove instances with missing features and anamolies. Title and text fields of each article were then concatenated to format data points into tuples containig content (title and text) and label, simplifying and speeding up processing in later stages. The content of each article was transformed into a vector of words by applying Spark's Tokenizer [] and sequence of tokens extracted by examining articles, stop words from NLTK's stop words module and punctuation marks from Python's string module were used to filter out noise and trivial tokens from the word vectors to allow for a more robust feature extraction by focusing on important words.

### Feature Extraction

To conevrt the tokenized and filtered articles to a set of features utilized by classification algorithms, term frequency inverse document frequency (TF-IDF) technique was employed. TF-IDF is a feature vectorization method widely used in text mining to reflect the importance of a term to a document in the corpus. Term frequency of ***t*** in document ***d*** denoted by ***TF(t, d)*** is the number of times that term t appears in document ***d***, while document frequency of term ***t*** in corpus ***D*** denoted by ***DF(t, D)*** is the number of documents that contain term ***t***. Using term frequency alone to measure the importance would lead to emphasizing terms that appear very often but carry little information about the document. To resolve this issue Inverse document frequency is used as a numerical measure of how much information a term provides.

<p alt="IDF" align="center"><a href="https://spark.apache.org/docs/latest/ml-features#tf-idf"><img src="https://github.com/rmanaem/veracity-detection/blob/master/figures/idf.png?raw=true"/></a></p>

<p alt="TFIDF" align="center"><a href="https://spark.apache.org/docs/latest/ml-features#tf-idf"><img src="https://github.com/rmanaem/veracity-detection/blob/master/figures/tfidf.png?raw=true"/></a></p>

Spark’s TF-IDF is implemented in two steps:

- HashingTF: Transformer that takes sets of terms and converts those into fixed-length feature vectors. It utilizes a hashing trick , mapping a raw feature into an index (term) by applying a hashing function. then term frequencies are calculated based on the mapped indices.
- IDF: Estimator which is fit on a dataset and produces an IDFModel. The IDFModel takes feature vectors produced by HashingTF and scales each feature [].

### Defining Training and Test Sets

Training and test sets were defined by k-fold cross validation implemented using Scikit-learn’s KFold. The data was split into 5 folds and models are trained and tested 5 times. In each iteration 4 folds are used to train and 1 was used to test the models. The average performance was computed to reflect the overall result.

### Classifiers

For the binary classification of the articles into True and False classes, K-nearest neighbors (kNN) and random forest (RF) classifiers were chosen. kNN was implemented using Scikit-learn’s KNeighborsClassifier[]. The number of nearest neighbor for each sample was determined by the square root of the input and incremented if the result was even. Random forest classifier was implemented using Scikit-learn’s RandomForestClassifier[]. Random forest parameters were tuned resulting in an ensemble of 45 trees with max depth of 20.

## Results

...

## Discussion

...

## References

[1] A. Bovet and H. A. Makse, “Influence of fake news in Twitter during the 2016 US presidential election,” Nature News, 02-Jan-2019. [Online]. Available: <https://www.nature.com/articles/s41467-018-07761-2>. [Accessed: 18-Feb-2021].

[2] J. A. Nasir, O. S. Khan, and I. Varlamis, “Fake news detection: A hybrid CNN-RNN based deep learning approach,” International Journal of Information Management Data Insights, 05-Jan-2021. [Online]. Available: <https://www.sciencedirect.com/science/article/pii/S2667096820300070>. [Accessed: 18-Feb-2021].

[3] I. Ahmad, M. Yousaf, S. Yousaf, and M. O. Ahmad, “Fake News Detection Using Machine Learning Ensemble Methods,” Complexity, 17-Oct-2020. [Online]. Available: <https://www.hindawi.com/journals/complexity/2020/8885861>. [Accessed: 18-Feb-2021].

[4] Brownlee, J. (2019, August 07). What is natural language processing? Retrieved February 19, 2021, from <https://machinelearningmastery.com/natural-language-processing/>

[5] Cheng, R. (2020, November 16). Top NLP libraries to Use 2020. Retrieved February 19, 2021, from <https://towardsdatascience.com/top-nlp-libraries-to-use-2020-4f700cdb841f>

[6] Scikit-learn. (n.d.). Retrieved February 19, 2021, from <https://scikit-learn.org/stable/>

[7] The beginner's guide to Text VECTORIZATION. (2017, September 21). Retrieved February 19, 2021, from <https://monkeylearn.com/blog/beginners-guide-text-vectorization/>

[8]Sklearn.feature_extraction.text.TfidfTransformer¶. (n.d.). Retrieved February 19, 2021, from <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#:~:text=The%20formula%20that%20is%20used,document%20frequency%20of%20t%3B%20the>

# Veracity Detection Using Supervised Learning

## Abstract

Dawn of the digital age and the World Wide Web paved the way for dissemination of high volumes of unrefined information through online media outlets and social media platforms. Ease of access, corporate and political biases, and lack of appropriate moderation in these mediums has resulted in propagation of false information and fabricated news, forging an essential demand for filtration. This project aims at analyzing related data using supervised learning tools and techniques to develop models for veracity detection. This work explores different textual properties and patterns that appear in news articles to extract linguistic features for training models using classification algorithms.

## Introduction

The advent of technology has surged through all aspect of life revolutionizing the way we receive, process, and pass on information. Multitude of our decisions are based on the information reported and broadcasted through different news and media outlets. These outlets and other entities utilize social media platforms (Facebook, Twitter, …) to further disseminate information to a broader audience. In a world where news organizations and the press have their own specific corporate and political allegiances and our means of communication and sharing information are designed to reinforce pre-existing biases, its essential to recognize the significance of validity of the available information. The 2016 US presidential election is widely used as the prime example to demonstrate the impact of disinformation on major events. Propagation of fake news through Facebook and Twitter in the months leading up to the election influenced voter turnout in democratic voters and voter decision in apolitical voters altering the course of the election [1].

The task of validating news as essential as it may be, can be time consuming and difficult even for experts as It requires exhaustive research into different aspects. To tackle these challenges, automatic detection models have been researched by employing artificial intelligence and machine learning principles. One of the recent works proposed a novel hybrid deep learning model that combines convolutional and recurrent neural networks for fabricated news classification. This model was able to successfully validate two fake news datasets and achieve results that were significantly better than other non-hybrid baseline methods [2]. Another study utilized machine learning ensemble approach to train a combination of different machine learning algorithms to expand their research beyond the domain of the existing literature. The ensemble learners showed promising results in a wide variety of applications as the learning models had the tendency to reduce error rate using techniques such as bagging and boosting [3].

This project is aimed at automating classification of news articles by training and testing machine learning models using supervised learning tools and techniques. This work explores key words, phrases, and other textual properties that appear in news articles to extract linguistic features for training models using classification algorithms.

## Materials and Methods

===new====
The dataset used in this project is the [Fake and real news dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset?select=True.csv) retrieved from Kaggle.com, a website containing a large collection of free-to-use Big Data datasets. The dataset is seperated into 2 files: fake.csv (51.2 MB), containing 23503 entries considered as 'fake' news and a slightly smaller true.csv (51.1 MB), containing 21418 entries that are accepted as being 'real' news. The total size of this dataset is 110 MB. Due to recency of the issue in question, the dataset needed to contain recent articles. In this dataset, the articles were dated throughout the span of the year of 2017. 
==new====

This project will use multiple technologies. First, Pandas will be used to convert the dataset into something that is easier to manipulate with code. The problem in this project requires Natural Language Processing (NLP). This project involves interpreting words into something that a machine can understand. There needs to be a way that can contextualize news article texts,  which can be done by NLP. Simply, NLP is defined as the automatic manipulation of natural language by software [4]. The library that will be used for NLP is nltk. Python's nltk is one of the most popular and widely accepted NLP library [5]. nltk's main objective in this project will be to pre-process the text. nltk provides powerful tools like tokenization, stemming, lemmatization, chunking and more to analyze the techs. Only some of the tools of nltk will be used in this project. Scikit learn will also be used in this project. Scikit learn is a Python ML library that contains features like classification, regression, clustering and more [5]. This library will be used for 2 important steps in this project: vectorization and ML algorithm.  Scikit learn contains  vectorization algorithms that helps convert words from tokenization to a data structure that can be processed by a machine. In other words, vectorization is converting tokens (array of words received from tokenization in the pre-processing step) of text into feature vectors which are numerical way of representing an object into a vector[7]. Multiple types of vectorization will be used including count vectorization and TF-IDF vectorization.

Since this project heavily relies on NLP, the approach that will be taken will be that of a NLP pipeline. This means that after the problem and dataset are selected and identified, the dataset will be pre-processed, vectorized and then put into ML algorithms. For pre-processing, the nltk library well be used. Although the data is cleaned, nltk will be used to pre-process to remove stopwords, which are words that do not add anything to the context or the sentence. Lemmatization will also be used in pre-processing. Lemmatization is defined as the grouping of words that have similar meaning together into one output (word in this case) to avoid reduntancy. It is especially useful since it allows similar words to be analyzed as one entity.  The data will also be tokenized by this library which will split the text into a list of strings in order to be computed much easily (mainly to be used in vectorization step). Scikit learn will then be used to vectorize the data so that we have a matrix in order to analyze our words. Count vectorization will create a matrix which returns the  the frequency of specific words in different articles. TF-IDF vectorization will also be used to check what is the importance of a word in a a particular string of text using the following formula:  tf-idf(t, d) = tf(t, d) * idf(t) where  idf(t) = log [ n / df(t) ] + 1 [8]. The Scikit learn provides with "Tfidvectorizer" which automatically does all the vectorization by itself. After this, the data will be split into training and test data using K-fold cross validation. Specifically, we decided to go with 5-fold cross validation. ML algorithms like decision trees and K-Nearest Neighbour (kNN) will then be used to classify the instances. Finally data will be analyzed by 3 metrics accuracy (number of correct prediction/observations), precision (number of real predicted positives/number of all predicted positives) and recall (number of real predicted positives/number of real postives).

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

# Veracity Detection Using Supervised Learning

## Abstract

Dawn of the digital age and the World Wide Web paved the way for dissemination of high volumes of unrefined information through online media outlets and social media platforms. Ease of access, corporate and political biases, and lack of appropriate moderation in these mediums has resulted in propagation of false information and fabricated news, forging an essential demand for filtration. This project aims at analyzing related data using supervised learning tools and techniques to develop models for veracity detection. This work explores different textual properties and patterns that appear in news articles to extract linguistic features for training models using classification algorithms.

## Introduction

The advent of technology has surged through all aspect of life revolutionizing the way we receive, process, and pass on information. Multitude of our decisions are based on the information reported and broadcasted through different news and media outlets. These outlets and other entities utilize social media platforms (Facebook, Twitter, …) to further disseminate information to a broader audience. In a world where news organizations and the press have their own specific corporate and political allegiances and our means of communication and sharing information are designed to reinforce pre-existing biases, its essential to recognize the significance of validity of the available information. The 2016 US presidential election is widely used as the prime example to demonstrate the impact of disinformation on major events. Propagation of fake news through Facebook and Twitter in the months leading up to the election influenced voter turnout in democratic voters and voter decision in apolitical voters altering the course of the election [1].

The task of validating news as essential as it may be, can be time consuming and difficult even for experts as It requires exhaustive research into different aspects. To tackle these challenges, automatic detection models have been researched by employing artificial intelligence and machine learning principles. One of the recent works proposed a novel hybrid deep learning model that combines convolutional and recurrent neural networks for fabricated news classification. This model was able to successfully validate two fake news datasets and achieve results that were significantly better than other non-hybrid baseline methods [2]. Another study utilized machine learning ensemble approach to train a combination of different machine learning algorithms to expand their research beyond the domain of the existing literature. The ensemble learners showed promising results in a wide variety of applications as the learning models had the tendency to reduce error rate using techniques such as bagging and boosting [3].

This project is aimed at automating classification of news articles by training and testing machine learning models using supervised learning tools and techniques. This work explores key words, phrases, and other textual properties that appear in news articles to extract linguistic features for training models using classification algorithms.

## Materials and Methods

The dataset consists of 2 .csv files: true.csv, containing real news and fake.csv containing fake news. Each datapoint contains 4 features: title, text, subject and date. The dataset is clean, does not have any missing values. For this missing values will not be checked in the pre-processing. The technique used to classify an instance will be supervised binary classification having labels true or false. 

The problem in this project requires Natural Language Processing (NLP). Because of this, Python's nltk will be used. Scikit learn will also be used in this project. The dataset will first be read by Pandas.

The nltk library will be used for multiple things. Although the data is cleaned, nltk will be used to pre-process to remove stopwords, which are words that do not add anything to the context or the sentence. Lemmatization will also be used in pre-processing. Lemmatization is defined as the grouping of similar words together to avoid reduntancy. It is especially useful since it allows similar words to be analyzed as one entity. 

Once the pre-processing is done, *Scikit learn will also be used to vectorize the data*. Scikit learn will also be used to split the data. In order to split the data, k-fold cross validation will be used instead of a random split, to get more accurate training. 

## Results

...

## Discussion

...

## References

[1] A. Bovet and H. A. Makse, “Influence of fake news in Twitter during the 2016 US presidential election,” Nature News, 02-Jan-2019. [Online]. Available: <https://www.nature.com/articles/s41467-018-07761-2>. [Accessed: 18-Feb-2021].

[2] J. A. Nasir, O. S. Khan, and I. Varlamis, “Fake news detection: A hybrid CNN-RNN based deep learning approach,” International Journal of Information Management Data Insights, 05-Jan-2021. [Online]. Available: <https://www.sciencedirect.com/science/article/pii/S2667096820300070>. [Accessed: 18-Feb-2021].

[3] I. Ahmad, M. Yousaf, S. Yousaf, and M. O. Ahmad, “Fake News Detection Using Machine Learning Ensemble Methods,” Complexity, 17-Oct-2020. [Online]. Available: <https://www.hindawi.com/journals/complexity/2020/8885861>. [Accessed: 18-Feb-2021].

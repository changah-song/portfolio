---
layout: post
title: Content-Based Recommenders (4/6)
description: “You like blue? Lemme introduce you to a slightly different shade of blue”
date:   2024-05-04 00:00:00 +0900
author: nolan
image:  '/images/rs-chapter4-thumbnail.avif'
tags:   [ML, NLP, Recommender System (추천 시스템), Python]
tags_color: '#477690'
---

### Overview
Content-Based Recommender Systems (CB RS) recommends items based on a **user’s historical item-rating data**. Looking at previous items the user favorably rated (either explicitly or implicitly), items similar to that one can be recommended. That is all. Here are the three steps that are followed:

1. Item profile. Also called content analyzer, attributes about the item are collected. In the context of movies, that could be the director, genre, length of film, keywords, etc. This step is the pre-processing step that extracts all relevant information from items. There are many feature extraction techniques available to use and has already been extensively researched in the field of **Information Retrieval**.
2. User profile. Also called profile learner, data representative of the user’s preferences are collected. Data is collected about the user’s historical ratings and is used to create a model that generalizes the users’ preference. This generalization step utilizes **Machine Learning** techniques.
3. Prediction. Also called filtering component, items that are similar to the user’s item preference are found and recommended. **Similarity metrics** are used to find the items most similar to what the user profile suggests that the user would like.

| **Note**: What is the difference between item-based CF and CB recommenders? I had this question pop up during my research and it seems like these two methods seem identical. Some resources describe them exactly the same way but the one explanation that made sense for me is the following: Item-based CF uses the items rated by various users as a basis to find similar users (to then recommend items that a similar user liked) while CB uses the profile of the items in a database and the items that a user liked to find similar items to then recommend. [1]

#### Strengths and Weaknesses
— — — — — — — — — — — — — — —

🔺 Strengths

- ***Other-User Independent***. The recommendations are not reliant on other users such as collaborative filtering (CF) since we just rely on the one user’s historical data to find similar items.

- ***Transparency***. Can explicitly let the user know why certain items are being recommended such as the content’s features or descriptions. “Since you liked action-packed, superhero movies such as ‘Avengers: Endgame’ and ‘Batman: The Dark Knight’, we think you will also like…”.

- ***New-Item Friendly***. New items can be recommended right away as long it is found to be similar to other items the user liked in the past. In CF, new items take time to be rated by other users and only when other similar users like it can it then be suggested to you.

🔻 Weaknesses

- ***Limited Content Information***. The most important step is to find all the relevant attributes of items. However, there is a natural limit to the number and type of features associated with items and oftentimes domain-specific knowledge is needed and that has to be manually added. When comparing to CF, in return for not needing other user’s data, it requires a lot of data on the items itself.

- ***Over-Specialization***. Limited ‘novelty’ and lack of ‘serendipity’. As discussed in the first post, novelty and more so serendipity are the goals of a good recommender system. Items recommended should be new (novel) but also unexpected (serendipitous). With the Content-based approach, the goal is to find items most similar to ones the user already liked and so, in theory, a ‘perfect’ CB recommender shouldn’t be recommending anything new while in practice, hardly any unexpected items are recommended.

- ***New-User Unfriendly***. Since this method depends on having historical data about the user, a new user without any prior ratings wouldn’t have any data to give to the recommender. This limits the effectiveness of this approach for new users and can also be seen as a type of ‘cold-start’ problem.

| **Note**: The strength and weaknesses of CF and CB are almost entirely flipped. CF struggles with relying on other users, weak transparency, and dealing with new items while it doesn’t deal with limited content information or over-specialization (it still struggles with the cold-start problem).

### Item Profile
Two steps: **Feature Extraction** and **Feature Similarity**

To get the item profile, we need feature extraction from items to convert them into a keyword-based vector-space representation. That is just a fancy way of saying that we need to collect information, whether it is structured or unstructured, about the item and map it into a vector space so that our algorithms can do stuff with it mathematically. Structured information is numerical or fields with few possible descriptions such as prices or colors. Unstructured information is usually text-based and describes the item in natural language such as a description of a book, its content, title, and author. Most content-based recommender use unstructured textual data as that is the easiest to find and lots of it can be found.

These features are then chosen to be included in the model depending on their relevance (**feature selection**) or included but with varying weights depending on their importance (**feature weighting**). Most content-based recommenders use a simple retrieval model called the Vector Space Model (VSM) with basic TF-IDF weighting.

| **Note**: We will focus on feature extraction from **text-based descriptions** of an item as that is the most common implementation of CB RS.

***How to extract features from item? … NLP***

**VSM** represents each item as a vector of term weights in n-dimensional space, each dimension being a term from the overall vocabulary of a given set of items. Well aren’t there so many unique words in a document? (Document will be used to mean a collection of texts). Correct ,and we will have each as a dimension. However, we can reduce the dimensions by utilizing some natural language processing (NLP) operations such as stop-word removal, stemming, and/or tokenization.
- Stop-word removal: Get rid of unnecessary or irrelevant words. For example, [a, an, the, is, was, etc.] in the English language.
- Stemming: Cut down words into its main component, usually by locating the root of the word or cutting out its prefix and/or suffix. For example, [buy, buying, bought] could all be reduced to [buy].
- Tokenization: Break a sequence of strings into groups whether it is by character, subword, word, phrase, or some other method. For example, “larger than life” can be tokenized word-wise as [“larger”, “than”, “life”], character-wise as [‘l’, ‘a’, ‘r’, ‘g’, ‘e’, ‘r’, …, ‘e’]

NLP is a large field in itself and so we won’t cover much more in detail but there are various ways to implement these methods and also other methods to extract features from text using NLP.

Once the document is sufficiently reduced and the terms are extracted, we should have a set of documents, D, and a set of terms, T. Each document in D is represented as a vector in a n-dimensional vector space, each with weights for each term in T.

<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-4-1.webp" loading="lazy" style="width: 700px;">
    </div>
    <p style="text-align: center; font-size: 15px; color: grey;">
        N is the number of documents and n is the number of total terms covering all the documents. w_kj is the weight for term t_k in document d_j
    </p>
</div>

***How to find the weights? … TF-IDF***

We know D (input) and T (NLP feature extraction). How do we calculate the weights for each document? We can use TF-IDF (term frequency-inverse document frequency) which is a method that finds relevant words in a document by multiplying two metrics: how many times a word appears in a document (TF), and the inverse document frequency of the word across a set of documents (IDF). This approach has its basis on three assumptions:

1. *TF assumption*: multiple occurences of a term in a document are not less relevant than single occurrences
2. *IDF assumption*: rare terms are not less relevant than frequent terms
3. *normalization assumption*: long documents are not preferred to short documents

> Terms that occur frequently in one document (TF) but rarely in the rest of the documents (IDF) are more likely to be relevant to the topic of the document. Normalizing prevents longer documents from having a better chance of retrieval.

*(Optional)* The math is as follow:

<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-4-2.webp" loading="lazy" style="width: 700px;">
    </div>
    <p style="text-align: center; font-size: 15px; color: grey;">
        TF-IDF = TF * IDF where TF is simply the frequency of a term k in document j and IDF is the inverse document frequency. Taken from Aggarwal (2016).
    </p>
</div>
The equation for TF is the frequency of a specific term, k, divided by the frequency of all terms, z. The IDF section just shows the log of the number of documents, N, over the number of documents where term k occurs at least once. This calculates the inverse document frequency. We multiply the two components and normalize it to get the weights, w_{k,j}, which is the last equation.

### User Profile
Learning user profiles is closely related to the classification and regression modeling problem as we’ve discussed in a previous post (Collaborative Filtering RS). When ratings are discrete values, the problem is text classification while when ratings are continuous, the problem is regression modeling. Given that a user rated certain documents, those documents are a set we can call D_L which is the training data and the un-rated documents we can call D_U. Each set has documents but the different is that D_L is rated or, in other words, labeled. We are just trying to label D_U and to do so we can use various classification and regression tactics.

The most simple, and one we are familiar with from CF recommenders is the nearest neighbor classification. There are other methods such as Bayes Classifier, Rule-Based Classifier, etc. but we will go with the simplest one.

We map D_L onto the same vector space as in the item profile step. The user profile is then a vector space with all the relevant documents they’ve rated. When we map the D_U documents onto that same space, we can use similarity metrics, such as cosine similarity, to find the nearest neighbors in that vector space.

### Prediction
Similar to CF recommenders, we can recommend items that were found to be similar to the items the user liked in their user profile. The simplest way would just be top-N recommendation where we pick the N most similar items to recommend.

### Visual Overview
<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-4-3.webp" loading="lazy" style="width: 700px;">
    </div>
    <p style="text-align: center; font-size: 15px; color: grey;">
        Visual overview of Content-Based RS. Image by Author.
    </p>
</div>
The basic steps in content-based is pretty straightforward and the image above shows the process we discussed above. We first start off with the user-item matrix and item-feature matrix. The item-feature matrix is where all the information from our feature extraction is stored in. Usually, features are not described in number (e.g. movie description, keywords, actors, etc.) and so we have to convert the feature information into usable, numerical values. This is where tokenization and other NLP methods can help.

Next, we create the user profile by combining the user-item matrix (ratings data) and the item-feature matrix (item profile). A simple matrix multiplication can do the job but there can be other approaches. Once the user-feature matrix is calculated, we can now know each users’ feature preferences.

Finally, we can compare the similarity between every item and the user’s feature preference. We find the top-n most similar items and recommend it to the user.

### Code Implementation
Let’s try implementing this is code. Three steps — (1) feature profile, (2) user profile, (3) recommendation based on similarity.

#### Import data and get feature profile
Using the MovieLens 1M dataset, we get the user-item matrix and the item-feature matrix. There is only one feature to consider for this dataset: genre. Since the genre is a categorical data type, we just need to replace each genre with a unique integer. That is what I do below from lines 18–23. Now we have the ratings matrix and the feature profile.

<script src="https://gist.github.com/paul-song-minerva/78ec68719832e19ea86e2b9c94bc6b27.js"></script>


#### User Profile and Recommendation
Now, we simply find the dot product between the two matrices to get the user-profile. What we end up getting is all the genres the users liked in the past and so we have a list of a bunch of genres and many are overlapping. Therefore, I replace the user preference with a dictionary that captures which genres they liked as a percentage. The output shows that for user 0, their top 3 genres were 10 (23.27%), 11 (16.63%), and 4 (10.45%). Given this, we can now recommend items that are in these genre categories.

<script src="https://gist.github.com/paul-song-minerva/edcf50d13810691cabb9dcf11628c6d6.js"></script>

```
{1: 0.028503562945368172,
 2: 0.0498812351543943,
 3: 0.0498812351543943,
 4: 0.10451306413301663,
 5: 0.007125890736342043,
 6: 0.030878859857482184,
 7: 0.0688836104513064,
 8: 0.035629453681710214,
 10: 0.2327790973871734,
 11: 0.166270783847981,
 12: 0.03800475059382423,
 13: 0.030878859857482184,
 14: 0.009501187648456057,
 15: 0.0332541567695962,
 16: 0.0688836104513064,
 17: 0.04513064133016627}
```

***
We have gone through the basics of content-based recommenders and an intuitive code implementation of each step. Now you should be able to understand more advanced extensions of this that will be discussed in chapter 6.

### Resources
Aggarwal, C. C. (2016). Recommender systems: The textbook. SPRINGER.

Kantor, P. B., Rokach, L., Shapira, B., & Ricci, F. (2011). Recommender Systems Handbook. Springer.
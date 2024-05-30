---
layout: post
title: Collaborative Filtering Recommenders (3/6)
description: “People similar to you liked this, try it out”
date:   2024-05-03 00:00:00 +0900
author: nolan
image:  '/images/rs-chapter3-thumbnail.avif'
tags:   [machine-learning, rec-sys]
tags_color: '#477690'
category: blog
---

### Overview
Collaborative Filtering methods relies on **other users** as well as the target user’s ratings to recommend relevant items. Think about how videos are recommended on Youtube. Usually once you start watching a video about the fall of Rome, you get recommended similar content, whether it be the subject (Roman history), video format (video essays), or content creator (same or other history-focused channels).

This is the basis of collaborative filtering and it makes sense intuitively. If A liked an item and A is similar to B, it follows that B would also like that item. Or in the same line of thought, if A liked an item, it follows that A would like similar items.

There are two main types of collaborative filtering recommender systems: memory-based and model-based.

This chapter will cover the very basics of these two categories to ensure that we have a firm grasp on the underlying concept and simple implementation. Later chapters will revisit CF but will explore more advanced ways to approach it.

***
## Memory-Based
This method uses the entire user-item ratings matrix to generate predictions. It uses statistical methods to search for a set of users who have similar transactions history to the active user. Such a simple concept translates to a simple implementation. We simply have to worry about three things: (1) method for calculating the similarity between users or items, (2) method for predicting what items a user might like, and (3) recommending items. Of course, these three key steps have various forms of implementation.

In memory-based RS, there are two categories: (1) **user-based** and (2) **item-based**. The difference is in how we calculate the similarities. The user-based approach uses *user similarity* to fill in the empty cells while the item-based approach uses the *item similarity* to do so. User-based is the more intuitive one that is largely used to explain collaborative filtering in general. Finding similar users and then ratings items that those similar users liked makes sense right away. On the other hand, finding similar items and then rating “users” that the items ended up being liked by is a bit harder to wrap your head around.

The visualization below shows the two different categories of memory-based RS. For the rest of this article, we will be using the user-based approach but the math and implementation behind item-based is almost identical. While reading the rest of the article, refer back to this visualization to make sure you know how we are moving from one step to the next.

<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-3-1.webp" loading="lazy" style="width: 700px;">
    </div>
    <p style="text-align: center; font-size: 15px; color: grey;">Two different approaches in memory-based recommenders.</p>

</div>

### (1) Similarity
| **Keywords**: Euclidean distance, Manhattan distance, Cosine similarity, Pearson correlation, KNN

Calculating similarity can be thought of as finding the distance between the two users. Why distance? As a simple example, imagine we want to compare the similarity between three people A, B, and C. We only know two things about each person — their height and weight. If we map out the “location” of these people in a space where the x- and y- axis are height and weight, we get the following graph:
<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-3-2.webp" loading="lazy" style="width: 700px;">
    </div>
    <p style="text-align: center; font-size: 15px; color: grey;">Visualization of users.</p>

</div>

We can see that A and B are closer in distance in this space and so we can conclude that they are more similar to each other than either are to C. This is the basis of calculating similarity: after mapping out each user in a space that captures the users’ features, we calculate the distance between our target user and the other users.

> After mapping out each user in a space that captures the users’ features, we calculate the distance between our target user and the other users.

The simplest way to calculate the distance here is the **Manhattan** and **Euclidean** distance. The following equation is for calculating the Euclidean distance where i is the number of dimension and x and y are the users. The Euclidean distance is essentially the Pythagorean equation when calculating the hypotenuse in a right triangle:
<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-3-3.webp" loading="lazy" style="width: 700px;">
    </div>
</div>


The closer a user is to another, the Euclidean/Manhattan distance should be closer to 0.

There are more ways to calculate the similarity amongst users that is commonly used: **cosine similarity** and **Pearson correlation**. If we imagine the space that the users are located in as a vector space (as a point can be represented as a vector), we get the following graph:
<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-3-3.webp" loading="lazy" style="width: 700px;">
    </div>
    <p style="text-align: center; font-size: 15px; color: grey;">Visualization of users in vector space.</p>
</div>


In simple terms, cosine similarity measures the angle between two vectors. If the angle is 0, the vectors overlap. This doesn’t take into account magnitude and so it isn’t really a ‘distance similarity’ but more of a ‘directional similarity.’ The following equation calculates cosine similarity.
<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-3-4.webp" loading="lazy" style="width: 700px;">
    </div>
</div>

Pearson correlation is very mathematically similar to cosine similarity. The details of Pearson correlation will not be discussed but if you are interested in the core difference between the two, <a href="https://leimao.github.io/blog/Cosine-Similarity-VS-Pearson-Correlation-Coefficient/">here’s</a> a nice resource that explains it briefly.

Our simple example above only considers two attributes of a person (height and weight). However, for a more accurate calculation of similarity, we might imagine adding more dimensions such as age and gender. As the dimensions increase, we are not able to visualize the location of each user in the higher dimensional space. This is not a problem when calculating the distance amongst users because the equations we use to calculate the distance can scale up to higher dimensions.

##### When to use Euclidean distance vs. Cosine similarity

When do we use one over the other? There are no strict rules that dictate which similarity functions are the best. Trial and error is helpful and sometimes what function you pick doesn’t matter too much. However, since we know the core difference (distance vs. angle difference), we can use the following heuristic:

* Cosine similarity when the magnitude of features doesn’t matter or even disrupt the outputs (for example, in finding similar text by word frequency). It is the similarity of ratio/scale.
* Euclidean distance when the magnitude is important (for example, if the factors have actual meaning). It is the similarity of actual values.

**Locating most similar users**
Now that we calculated the similarity value amongst users, we simply need to find the most similar users and then we can move onto the prediction phase. A very commonly used method is the **K-nearest neighbor (KNN)**. As the name suggests, it uses similarity functions (as discussed above) to find the K-nearest neighbors. It has essentially two parameters: the similarity function and k-value. The similarity function is what we discussed above and the k-value just picks the top-k most similar users.

### (2) Prediction
After the closest neighbors have been located, we should now fill in the empty cells in our user-item ratings matrix. Since we know the target user’s similarity with all other users in the database, we can do some more calculations to guess, or predict, the values of the empty cells.

The simplest way to approach this is the weighted average method.
<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-3-5.webp" loading="lazy" style="width: 700px;">
    </div>
    <p style="text-align: center; font-size: 15px; color: grey;">Weighted average. Image by Abjiheet Anand (2020).</p>
</div>

This looks complicated but it is essentially taking the weighted average of the neighbors’ ratings (decided by KNN) for an item that the target user hasn’t rated yet and adding to the target user’s average rating.

### (3) Recommendation
Now that we filled out the empty cells in the user-item ratings matrix for our target user, we can simply recommend the items with the highest predicted rating scores.

#### Code Implementation
Let’s translate theory into code. We are using the MovieLens 1M dataset. There are plenty of libraries that can implement this faster but we will go through this process from scratch so that we can see clearly what is going on. First, we load the data and clean it. Below, I make sure that the index of the users and items are consecutive (pd.factorize) and normalize the user-item matrix so that variances in user’s rating behavior are accounted for.

<script src="https://gist.github.com/paul-song-minerva/da2e6f22be80e7293427bc6ee83d7edb.js"></script>

Next, we find the similarity between the users and the items and so we are trying out both the user-based and item-based approach. I’m also using cosine similarity and pearson similarity to see if there is a large difference.

<script src="https://gist.github.com/paul-song-minerva/da60c6340b0ba6880bd3f586912c9a86.js"></script>

Then, we use the similarity matrices to predict empty cells using a simple weighted average of the top-n similar users/items to the target user/item.

<script src="https://gist.github.com/paul-song-minerva/67dab0afdaa07df075f010e3a5b4ad93.js"></script>

Finally, we can evaluate the performance of the four approaches using three metrics: RMSE, MAE, and Spearman’s coefficient. From the previous chapter, you should be familiar with these evaluation metrics. If not, go here and catch up!

<script src="https://gist.github.com/paul-song-minerva/93095deda3f2e87fb3c1fe69a62432ac.js"></script>

```
User-based approach with cosine similarity
RMSE:  0.26672632258637224 
MAE:  0.0943938468652809 
Spearman:  0.5938959359206174
None 
---------
User-based approach with pearson similarity
RMSE:  0.2667263225863723 
MAE:  0.09439384686528093 
Spearman:  0.5938959349277827
None
---------
Item-based approach with cosine similarity
RMSE:  0.28283421661104957 
MAE:  0.09080784956365832 
Spearman:  0.23046860395364357
None 
---------
Item-based approach with pearson similarity
RMSE:  0.2734087079745447 
MAE:  0.08185400557506971 
Spearman:  0.24259847846933547
None 
---------
```

The rating accuracy metrics (RMSE and MAE) are similar for all of the approaches but we can see a clear difference in the Spearman correlation which shows how accurate the predicted ranking was to the true ranking. The user-based approach outperformed the item-based approach by far and this makes sense as we had a lot more users (~6000) than items (~3000) in our dataset.

This shows that when we have more users than items, user-based CF RS are preferable while when we have more items than users (which is usually the case), item-based CF RS are better.

***
## Model-Based
This method creates a model from the user rating data and uses it to recommend items. It reduces or compresses a large but sparse item-user matrix to improve performance and makes and uses a model to make predictions. This approach utilizes Machine Learning and Data Mining concepts such as classification, clustering, and rule-based approaches.

| **Note**: The distinction between memory- and model-based CF is quite arbitrary since memory-based KNN recommenders can technically be classified as a model (although it is a lazy learner model).

The most concise way I’ve found model-based CF to be divided is in three categories: non-parametric approach, matrix factorization-based, and deep learning. As mentioned above, the non-parametric approach (such as KNN from above) are similar to memory-based CF. We will explore this category in more detail in the future but for now, we will leave clustering based algorithms there.

The other two main categories are **Matrix Factorization-based** algorithms and **Deep Learning** algorithms. Deep Learning is a bit more advanced and so we will discuss it in a future chapter. Just understanding Matrix Factorization-based algorithms is good enough for a basic overview of model-based recommender systems.

Before we move on, let’s see why a model-based algorithm might be preferable.

**Advantages of model-based CF**

Taken from Aggarwal, 2016:

| 1. Space-efficiency: Typically, the size of the learned model is much smaller than the original ratings matrix. Thus, the space requirements are often quite low. On the other hand, a user-based neighborhood method might have O(m2) space complexity, where m is the number of users. An item-based method will have O(n2) space complexity. 
| 2. Training speed and prediction speed: One problem with neighborhood-based methods is that the pre-processing stage is quadratic in either the number of users or the number of items. Model-based systems are usually much faster in the preprocessing phase of constructing the trained model. In most cases, the compact and summarized model can be used to make predictions efficiently.
| 3. Avoiding overfitting: Overfitting is a serious problem in many machine learning algorithms, in which the prediction is overly influenced by random artifacts in the data. This problem is also encountered in classification and regression models. The sum arization approach of model-based methods can often help in avoiding overfitting. Furthermore, regularization methods can be used to make these models robust.

#### (1) Matrix Factorization
| **Keywords**: Singular Value Decomposition, Stochastic Gradient Descent

Matrix factorization aims to reduce the item-user matrix and decompose it into smaller parts. This allows for the attributes and preferences of users to be determined by a small number of hidden factors. Once decomposed sufficiently, we can use these decomposed parts to fill in missing ratings. This will be clearer as we go on. One such popular technique we will focus on here is Singular Value Decomposition (SVD).

**SVD (Singular Value Decomposition)**
Given our user-item ratings matrix of A with m users and n items, our model aims to learn:

* a user embedding matrix U with m x i dimensions
* an item embedding matrix V, with n x j dimensions

<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-3-6.webp" loading="lazy" style="width: 700px;">
    </div>
    <p style="text-align: center; font-size: 15px; color: grey;">Matrix Factorization. Image by Google </p>
</div>

The image above shows the decomposition of our matrix A into two smaller matrices U and V. The goal of SVD is to learn the decomposed matrices so that UV^T is a good approximation of A. That’s it.

> The goal of SVD is to best learn the decomposed matrices so that it is a good approximation of A when combined.

After the two decomposed matrices are found/trained, we can go back and fill in the blanks from the original ratings matrix. That is how we predict and recommend items to users.

### Objective Function

To find the matrices U and V, we treat this like an optimization problem. We start with a bad formation of these matrices and calculate the error. We iteratively update these matrices to minimize the errors.

One common and intuitive way to approach the objective function is by using squared distances. We minimize the sum of squared errors over all pairs of observed entries:
<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-3-7.webp" loading="lazy" style="width: 700px;">
    </div>
</div>

A very popular and basic optimization method is **Stochastic Gradient Descent**. If you are not familiar with this concept, I recommend you find materials dedicated to it specifically . However, to summarize briefly, it is an iterative optimization technique that follows the steepest “direction down”. We know from calculus that the slope of a function is its derivative and if it is positive, the function is going up and vice versa. With simple functions, we can simply calculate the derivative of a function and look for where it is zero to locate its local minimum or maximum. However, with complex (non-convex) equations where derivatives are more complicated, we have to take it slower and iteratively go down the steepest slope to reach an approximation of a local minima.

After we apply gradient descent to the objective function and minimize the loss, we can be sure that our decomposed U and V matrices can be matrix multiplied to approximate the original ratings matrix, A. With the decomposed matrices, we can then fill in the missing cells in A.

**Code Implementation**
The Surprise library has many methods that can help us easily implement recommenders. We use the SVD method in this library to quickly and easily implement a model-based CF with the same data as the code implementation of the memory-based CF from above.

<script src="https://gist.github.com/paul-song-minerva/80f44a4b9a25e14da656a7b3f407e30b.js"></script>

```
RMSE: 0.8718
MAE:  0.6851
```

One downside of using Surprise is that there are not many evaluation metrics we can use. RMSE and MAE are readily available but the other evaluation metrics discussed in the previous chapters are no where to be found. Therefore, I was only able to evaluate this model based on its RMSE and MAE score.

Something odd you might notice is that the memory-based model seems to outperform the SVD approach by far. The closer the scores are to zero, the more accurate the ratings are. This might be because of various reasons. First, the memory-based approach normalized the user-item matrix and thus reduced a lot of noise that can come from the rating patterns of each user. Second, the memory-based approach filled in all empty cells with 2.5 (a neutral rating) but that might have lead to an inaccurate prediction. However, we also know that the Spearman’s coefficient was not bad for the user-based approach and so if we calculate the Spearman’s coefficient for the SVD approach, perhaps we can have a fuller comparison of these two approaches.

***
That was the basics of Collaborative Filtering recommendation systems. Now that you know the underlying concepts of both, we can explore more advanced ways to approach these methods in the future.

### Resources
Aggarwal, C. C. (2016). Recommender Systems. <a href="https://doi.org/10.1007/978-3-319-29659-3">https://doi.org/10.1007/978-3-319-29659-3</a>

Anand, A. (2020, October 3). User-user collaborative filtering for jokes recommendation. Medium. Retrieved from <a href="https://towardsdatascience.com/user-user-collaborative-filtering-for-jokes-recommendation-b6b1e4ec8642">https://towardsdatascience.com/user-user-collaborative-filtering-for-jokes-recommendation-b6b1e4ec8642</a>

Cheng, Z. (2022, June 29). A comparison of cosine similarity vs euclidean distance in ALS recommendation engine. Medium. Retrieved from <a href="https://medium.com/nerd-for-tech/a-comparison-of-cosine-similarity-vs-euclidean-distance-in-als-recommendation-engine-51898f9025e7#:~:text=According%20to%20my%20research%2C%20it's,the%20factors%20have%20actual%20meaning).">https://medium.com/nerd-for-tech/a-comparison-of-cosine-similarity-vs-euclidean-distance-in-als-recommendation-engine-51898f9025e7#:~:text=According%20to%20my%20research%2C%20it's,the%20factors%20have%20actual%20meaning).</a>

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=<a href="http://dx.doi.org/10.1145/2827872">http://dx.doi.org/10.1145/2827872</a>

Google. (n.d.). Matrix factorization. Google. Retrieved from <a href="https://developers.google.com/machine-learning/recommendation/collaborative/matrix">https://developers.google.com/machine-learning/recommendation/collaborative/matrix</a>

Grover, P. (2020, March 31). Various implementations of collaborative filtering. Medium. Retrieved from <a href="https://towardsdatascience.com/various-implementations-of-collaborative-filtering-100385c6dfe0">https://towardsdatascience.com/various-implementations-of-collaborative-filtering-100385c6dfe0</a>

Mao, L. (2021, September 22). Cosine similarity vs Pearson correlation coefficient. Lei Mao’s Log Book. Retrieved from <a href="https://leimao.github.io/blog/Cosine-Similarity-VS-Pearson-Correlation-Coefficient/">https://leimao.github.io/blog/Cosine-Similarity-VS-Pearson-Correlation-Coefficient/</a>
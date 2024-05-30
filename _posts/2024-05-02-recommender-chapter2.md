---
layout: post
title: Evaluating Recommender Systems (2/6)
description: How to approach evaluating recommenders. It’s more complicated than you might think
date:   2024-05-02 00:00:00 +0900
author: nolan
image:  '/images/rs-chapter2-thumbnail.avif'
tags:   [machine-learning, rec-sys, python]
tags_color: '#477690'
---

> LEARNING GOALS — 4 different ways to evaluate recommender systems: accuracy, correlation, utility, and usage.

Before we start exploring the many different ways to implement recommender systems, we should first discuss how we can evaluate them because only then can we we compare different algorithms and see which one performs better in which ways. However, even before evaluating specific recommenders, let’s discuss two broad situations we find ourselves in when evaluating recommenders: offline and online.

| **Note**: The offline approach is mostly used in evaluating recommenders and the one I’ll be using for the rest of the series but I discuss online approach for completeness.

#### Online

When evaluating how well our recommender performs online, we are measuring the **users’ (usually real-time) reactions** given the recommended item. For example, the clicks or views a recommended Youtube video gets gives insight into how effective the recommendation was. Some methods used in this approach is **A/B testing** and **multi-armed bandit** recommenders. 

Briefly explained, A/B testing aims to see how well a different course of action impacts user behavior by conducting both the default (A) and new (B) at the same time and seeing which performs better. 

For example, to user 1, I can recommend an item A and then to another similar user 2, I can recommend an item B. If item B ends up getting more clicks/purchases, we can conclude that recommending B was more effective. 

A more nuanced approach to this is the multi-armed bandit approach where we optimize the recommendations by working within the tradeoff between exploration and exploitation After recommending various items (exploration), we get a sense of what is liked and so we can focus on recommending that item (exploitation). This is especially useful in combating the Cold-Start problem where there is initially no data to make good recommendations right away.

However, the online evaluation approach involves active user participation and an evaluation in real time and so we will not be using this approach in this series.

#### Offline

From a research and practice perspective, we use the offline approach as we have access to many historical data and various data types that can lend to a better assessment of the generalizability strength of our recommenders. To start, we have to know that there is not one full-proof way to evaluate the effectiveness of recommenders as it has various goals as discussed in an earlier chapter.

There are two ways to evaluate recommenders — **rating** or **ranking**. Rating evaluates how accurately the recommender predicts the ratings of items. Ranking evaluates how well the recommender recommends items that is desirable to the user. Below are four common evaluation methods that either evaluate rating or ranking.

### Accuracy (RMSE) — rating
This is the most straightforward method. We look at the predictions and compare that to the truth value. This means that when we train our model, we usually want to leave out some items that the user already rated, also known as the train-test split approach, so that we can compare prediction to truth. If this is done for all the users in our data and find the average, we can get a score for how accurately the recommender predicted ratings.

A common approach used in regression problems is ***mean squared error (MSE)***, ***root mean squared error (RMSE)***, and ***mean absolute error (MAE)***. These are all simple ways to evaluate the accuracy of a classification algorithm. For recommenders, we have predicted ratings and truth ratings and so we can apply any of these methods to get a score.

RMSE measures the *average difference between values predicted by a model and the actual values*. This leads to a simple formula and its translation into Python code shown below assuming ‘pred’ holds the predicted ratings and ‘true’ holds the true ratings 💻.


<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-2-eq1.webp" loading="lazy" style="width: 700px; height: 100px;">
    </div>
</div>

```python
import numpy as np
rmse = np.sqrt(np.mean((pred - true)**2))
```

| **Note**: This approach penalizes deviances that go further away from the truth value by squaring the difference between pred and true. MAE simply takes the absolute value of the difference instead of squaring it. Which is better has no clear answer and is a topic that the reader can choose to research further.

### Correlation (Spearman) — ranking
How can we evaluate recommenders without looking at the rating accuracy? Look at the ranking accuracy! This approach measures how effective the top-n recommendations are. We just want to check if our ranking is similar to the true ranking. One approach to this is to use the rank correlation coefficient and the main implementation of this is by using the Spearman rank correlation coefficient (though there are others such as the Kendall rank correlation coefficient with its own pros and cons).

*Spearman rank correlation coefficient* approach goes as follows:

1. Rank all items for the prediction and truth value for a user
2. Apply Pearson correlation coefficient between pred and true rankings
3. Average over all users to obtain global correlation coefficient

The coefficient will be in range (-1, +1) and large positive values are more desirable as the ranking of the predicted more highly correlated with the true ranking.

Using the *scipy.stats* library and its *spearmanr* method, one way we can implement this is in Python is shown below. This algorithm assumes that ‘true’ holds the true rating of every user and ‘pred’ holds the predicted ratings of every user (*np.argsort* finds rankings based on the ratings) 💻.

```python
from scipy.stats import spearmanr
spearman_scores = []
for user in range(len(true)):
    true_rank = np.argsort(true[user])
    pred_rank = np.argsort(pred[user])
    spearman_scores.append(spearmanr(true_rank, pred_rank).correlation)
average_spearman = np.mean(spearman_scores)
```

### Utility (R-Score and ARHR) — rating & ranking
Realistically, we want to consider **both the rating and ranking** when evaluating recommenders. Accuracy measures, such as RMSE, are useful but don’t take into account the difference in importance of items highly ranked vs. lowly ranked. High ranked items are what the user is actually going to see. 

The utility approach aims to use both the rating and ranking to evaluate the effectiveness of recommenders and assumes two things: (1) higher rated items are of greater utility and (2) higher ranked items are of greater utility. An intuitive way to implement these rules is make it so that our evaluation value is lower when the rank-based utility is higher (further down the recommended list) and higher when the rating-based utility is higher: *rating-utility / ranking-utility*.

* **R-Score**

One approach is the **R-score** where the utility, U(u, i), of item i to user u is a product of the rating-based and ranking-based utility values as shown below. This utility function finds the utility of all items for a user and then the average of that is found to get the R-score value of a recommender.

<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-2-eq2.webp" loading="lazy" style="width: 700px; height: 100px;">
    </div>
</div>
where v_i is the ranking of the item i and alpha is a half-life parameter

<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-2-eq3.webp" loading="lazy" style="width: 700px; height: 100px;">
    </div>
</div>
L limits the amount of ranked items we use in our calculation. For example, we can choose to just use the top-10 (L=10) items to calculate the ranking-based utility.

<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-2-eq4.webp" loading="lazy" style="width: 700px; height: 100px;">
    </div>
</div>

| **Note**: Looking at the denominator when calculating U, we can see that this approach penalizes low-ranking items a lot. In situations where ranking doesn’t matter too much (such as news recommendations), we can choose to lighten the penalty. One such approach is the discounted cumulative gain (DCG).

* **ARHR**

Another approach is the *average reciprocal hit rate (ARHR)* also called *mean reciprocal hit-rate (MRR)*. This approaches is designed for implicit feedback data. Implicit feedback is given whenever a user clicks on an article, completes a video, or buys an item. The user is not explicitly rating items but their behavior can implicitly tell us what they like. This means that the data will likely be either 1 (bought, liked, clicked) or 0 (didn’t). While it is designed for implicit feedback where missing values can be set to 0, it can be generalized to be used for explicit feedback.

In this context, we can apply the same principle as above where the rank-based utility is 1/v_i (where v_i is the rank of item i) and the rating-based utility is simply r_ui.

<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-2-eq5.webp" loading="lazy" style="width: 700px; height: 100px;">
    </div>
</div>
<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-2-eq6.webp" loading="lazy" style="width: 700px; height: 100px;">
    </div>
</div>


### Usage (Precision, Recall, and ROC)
* **Precision & Recall**

In the context of recommenders, there are four scenarios.

* **TP, true-positives**: relevant items + recommended
* **FP, false-positives**: irrelevant items + recommended
* **FN, false-negatives**: relevant items + not recommended
* **TN, true-negatives**: irrelevant items + not recommended

*Precision* is defined as the percentage of recommended items that turn out to be relevant. All recommended items is TP + FP and relevant and recommended is TP and so the formula for precision is TP/(TP + FP).

*Recall* is defined as the percentage of relevant items that have been recommended. All relevant items is TP + TN and relevant and recommended is once again TP and so the formula for recall is TP/(TP+TN).

<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-2-eq7.webp" loading="lazy" style="width: 700px; height: 100px;">
    </div>
</div>


One way to unify the two metrics is the F1-measure which is the harmonic mean between the precision and the recall. The closer the precision, recall, or F1 are to 1, the better the recommender.

<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-2-eq8.webp" loading="lazy" style="width: 700px; height: 100px;">
    </div>
</div>
F_1 measure that unifies precision and recall.

* **ROC**
If you’re familiar with the ROC (receiver operating characteristic) curve, you would know that is a graph showing the performance of a classification model at all classification thresholds. It uses binary data and so is fit for evaluating recommenders that use binary data (usually implicit feedback). The x-axis is the false positive rate (FPR) and the y-axis is the true positive rate (TPR).

<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-2-eq9.webp" loading="lazy" style="width: 700px; height: 100px;">
    </div>
</div>

If the recommended list is too small, recommenders will not recommend all relevant items (false negative) and if the list is too large, recommenders will start to recommend irrelevant items (false positive). This trade-off can be captured in an ROC curve and we can quantify how effective the recommender is at recommending relevant items that has a high chance of being used.

***
We’ve seen how there are many different ways to evaluate the effectiveness of recommenders. From looking at ratings, rankings, and both and considering how much the ranking order matters to which type of data we are working with (implicit or explicit, binary or multivariate), the evaluation metric can change. The best evaluation metric depends on the recommender we’re implementing and so when trying to find the best evaluation metric for your recommender, I hope that this guide can be a good starting point where you can now confidently find more nuanced metrics on your own!

#### Resources

Aggarwal, C. C. (2016). Recommender Systems. <a href="https://doi.org/10.1007/978-3-319-29659-3">https://doi.org/10.1007/978-3-319-29659-3</a>
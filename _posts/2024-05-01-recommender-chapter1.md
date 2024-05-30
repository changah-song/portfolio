---
layout: post
title:  Recommendation System for Dummies (1/6)
description:
date:   2024-05-01 00:00:00 +0900
author: nolan
image:  '/images/rs-chapter1-thumbnail.avif'
tags:   [machine-learning, rec-sys, python]
tags_color: '#477690'
---

### 1. What are Recommender Systems?
Hello there, if you are someone with a very basic knowledge of AI and ML concepts and want to explore a very practical application of these subjects, recommender systems are a great place to start as they employ so many of the theoretical concepts covered in classroom settings.

What are recommender systems? As its name suggests, it is any system put in place that tries to recommend the best (in whatever metric that is defined) item to a user. Here, item can be anything from books or movies to friend suggestions; it is just anything that can be recommended to a user.

Recommender systems have been around for a while now but is a constantly evolving field as our knowledge of new machine learning and optimization techniques grows. The best recommender systems are what Netflix uses to suggest a new movie you might enjoy or what Youtube uses to keep you hooked for hours at a time.

There are many resources online, from other Medium articles to textbooks to Google courses, that explain many different types of recommender systems. So then why am I writing this series on recommender systems? It is mainly to have a centralized place where all the different types of systems can be discussed where the tone and structure of each post is familiar. There are many different types of recommender systems and not one centralized place where are all of it is discussed; the resources you can find on each different systems varies greatly and I would like to create a resource that introduces each one in one place.

The best centralized resource I could find were the two following textbooks: “Recommender Systems: The Textbook” by Charu C. Aggarwal (2016) and “Recommender Systems Handbook” by Francesco Ricci et al. (2011). These textbooks has most of the currently known recommender systems but is lengthy and has no code implementation. As such, my mission was to summarize all of the concepts discussed in these resources, with the help of other online resources, and to also include code implementation for better understanding.

### 2. Overview of Series

<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-chapter1-overview.png" loading="lazy" style="width: 800px; height: 300px;">
    </div>
</div>

Overview of the series and the four main categories of recommender systems. Image by author.
This series will initially cover the four broad categories of recommender systems: Collaborative Filtering, Content-Based, Knowledge-Based, and Hybrid Recommenders. These four categories alone cover most of what makes up recommender systems. Additional chapters to this series will be less-used, but still useful, systems or more advanced explorations of systems already covered earlier. We will also be covering case-studies, equipped with our new-found knowledge.

A brief summary of the four main categories are as follows:

* **Collaborative Filtering (CF)** — Uses the ratings of other users to recommend items to you.
* **Content-Based (CB)** — Uses your own historical data to recommend items to you.
* **Knowledge-Based (KB)** — Uses your explicitly requirements to recommend items to you.
* **Hybrid** — Combines algorithms in various ways to create a more well-rounded recommender.

While seemingly simple, there is a lot of information to cover in each of these fields. Anything that can be optimized, will be, and there are a lot of trade-offs to be considered. Through the discussion of these three categories, you should get a feel for the advantages and disadvantages of each and what we look out for in recommender systems.

### 3. Keywords to Familiarize
When diving into the field of recommender systems and especially if you want to do your own research and read up on more information during and after this series, there are keywords that will be very helpful if you familiarize yourself with them. These are mainly the goals and challenges of recommender systems. No matter what recommender system you choose to research, many of these keywords will pop up. I believe it will be helpful if you can greet most of those concepts here before moving on.

However, the first concept you should be familiar with is the **user-item ratings matrix**. In its simplest form, imagine a table where the row heading is the item (i.e. Avengers, Lion King, Barbie, etc.) and the column heading is the user (i.e. Adam, Bob, Christine, etc.). The cells where the item and user intersect are the ratings given by the user for the item. Now, viewed as a matrix, it is a m x n matrix containing m users and n items.

#### Goals

The goals of a recommender system are usually the same. Different systems try to reach these goals in various ways and how it does so will be covered in each chapter.

* **Relevance**. The most obvious one is relevance. Are the items being recommended relevant to the user? This is usually necessary in order for the user to engage with the item.
* **Novelty**. The item recommended should be new to the user.
* **Serendipity**. The item recommended should be unexpected to the user. This is slightly different from novelty as even new items can be expected. For example, if I know that I enjoyed Avatar, being recommended Avatar 2 is not unexpected although it is novel.
* **Diversity**. The items recommended should be diverse so that there is a higher change of the user liking at least one of what was recommended.

#### Challenges

Throughout the series, the most common concepts that will be reoccurring are the challenges that recommender systems face in general and how different systems aim to combat these different challenges.

* **Long-Tail**. Only a fraction of items are rated frequently. Why is this a problem? (1) High-frequency items are relatively competitive with little profit for the merchant. There is more profit to be gained from the lower frequency items. (2) Difficult to provide robust rating predictions as many recommender systems tend to recommend popular items rather than infrequent items. Has a negative impact on diversity.
* **Data Sparsity**. The user-item ratings matrix is typically not going to be full. There wil be missing ratings as many users have not rated or have not interacted with many items. This leads to a matrix that is sparse and thus leads to less accurate predictions and recommendations.
* **Scalability**. A recommender system feasible at a small scale might be infeasible at a larger scale. The time and space complexity of the algorithms used must be carefully considered.
* **Cold-Start**. New items cannot be recommended until some users rate it, and new users are unlikely to be given good recommendations because of the lack of their rating or purchase history. When the number of initially available ratings are relatively small, it can be difficult to give recommendations because there is no way to compare a new user (that we want to recommend items to) with another user or look at their historical data.
* **Curse of Dimensionality**. When working with data in high-dimensional spaces, it can be challenging to analyze the data and make predictions with it. This isn’t an issue when just looking at user, item, and ratings but in many recommender systems, we want to include even more information such as age, location, gender, etc. for a more robust prediction. However, handling high-dimensional data is computationally more expensive and more complex.

### 4. Dataset
The dataset that will be used for most of the code implementations is the MovieLense 1M dataset. This is a staple in many recommender system implementation resources you will find online. This is a simple dataset that contains the ratings of 6000 users for 4000 movies.

Download the dataset and save it for future use.

***

Are you ready to explore the many different types of recommender systems? Let’s go!
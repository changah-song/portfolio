---
layout: post
title: Knowledge-Based Recommenders (5/6)
description: “Ask and you shall receive”
date:   2024-05-05 00:00:00 +0900
author: nolan
image:  '/images/rs-chapter5-thumbnail.avif'
tags:   [machine-learning, rec-sys]
tags_color: '#477690'
category: blog
---

### Overview
Knowledge Based Recommender Systems (KB RS) is the last (out of three) broad category of recommender we will discuss. Knowledge based recommenders specialize in receiving and utilizing explicit user requirements and using those to recommend items. So far, with both Collaborative Filtering and Content-Based recommenders, we had to predict what the user would like based on their rating information. The data we used was either explicit (e.g. likes on an Instagram post) or implicit (e.g. watch time on Youtube videos). However, this approach takes the explicit data step a lot further and **specifically asks the user what they would like to find**. Pretty simple huh?

There are two types of KB RS:

- **Constraint-Based**: Given user requirements, this approach uses predefined recommender knowledge bases that contain *explicit rules* about how to relate customer requirements with item features. This is pretty much an *expert system* in **knowledge base systems** and is a constraint satisfaction problem.
- **Case-Based**: A user picks a specific item (case) as a target or anchor point and the algorithm finds a **similar item to recommend**. Results are usually used as new target cases with some interactive modifications. You’ll notice that this sounds similar to *Content-Based recommenders* where items similar to ones the user previously liked are suggested. The main difference is that most KB RS depend on the description of the items in the form of **relational attributes in knowledge bases** rather than as text keywords like in CB RS.
Although quite different, both are considered KB algorithms because they encode various types of domain knowledge in the form of constraints, rules (constraint-based), similarity metrics, and utility functions (case-based) during the search process.

#### Strengths and Weaknesses
— — — — — — — — — — — — — — —

🔺 Strengths

- ***Complex Item Domain***. When the item has many complex aspects to consider and requires expert knowledge in the domain, KB RS are able to capture that using knowledge bases curated by these experts. This is especially important when the item is of great significance and requires a lot of thought such as the purchase of a house.

- ***Rarely-Bought Items***. Items that are not frequently bought doesn’t garner enough ratings for CF & CB recommenders to be able to work.

- ***Avoids Cold Start Problem***. User data is not needed as what the user wants is explicitly defined. Recommendation can be accurate and start as soon as the user wants without the recommender requiring existing rating data to work well. This is a great strength over CF & CB recommenders.

🔻 Weaknesses

- ***Knowledge Acquisition Bottleneck***. The creation of the knowledge base requires the conversion of the knowledge possessed by domain experts into formal, executable representations

| **Note**: The main takeaway should be that the three main categories of recommenders are good for different contexts. For KB, we don’t care too much for serendipitous recommendations, as we did for CF and CB, because the user is looking for something specific rather than something unexpected but pleasant.

### Constraint-Based
The following section will go into the theory and logic underlying this method but it is not necessary to understand to know what the constraint-based KB algorithm does. Read if you want to know its base level ties to knowledge bases and finite state machines.

| **Note**: Knowledge base systems are an entire field on its own in AI and so I suggest finding resources to familiarize yourself with this topic if you haven’t learned about it. Briefly put, as shown in the image below, KB systems require the creation of a knowledge base (an organized collection of facts about the domain) and the rules engine (a bunch of logical rules, usually if-else statements, that represents the domain knowledge) by an expert in the domain. When the non-expert user specifies explicit requirements, the KB system runs it thorgh the rules engine to find something in the knowledge base that lies under the constraints of what the user wanted.

<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-5-1.webp" loading="lazy" style="width: 700px;">
    </div>
    <p style="text-align: center; font-size: 15px; color: grey;">
        Source: Javapoint, N.D.
    </p>
</div>
Ricci et al. defines two variables and three sets of constraints. The two variables are the user and item attributes (V_C, V_PROD) while the three constraints are logical constraints, user-item constraints, and allowed recommendation constraints (C_R, C_F, C_PROD). This becomes a constraint satisfaction problem where we have to find the right variables to fit all the constraints.

An example of a knowledge base with these variables and constraints is shown below:
<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-5-2.webp" loading="lazy" style="width: 700px;">
    </div>
    <p style="text-align: center; font-size: 15px; color: grey;">
        Example of KB in financial services. Image from Ricci et al. (2011)
    </p>
</div>

Given concrete user requirements, C_C, the solution cannot violate the following logic:
<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-5-3.webp" loading="lazy" style="width: 700px;">
    </div>
</div>

As long as an item meets this condition, it can be recommended to the user. One such way of doing that is in the form of finite state models (FSM). Finite state models can represent all the relationships built up in the KB in a graphical way. As long as an item can pass through all the nodes and reach the end, it satisfies all the constraints and can be recommended.

| **Note**: Finite State Models is a topic in the theory of computation that is heavily based on logic and can be used to graphically represent relationships between many different states as long as the rules for the relationships are well defined. It is a fascinating subject but one you don’t need a deep understanding of to understand Constraint-Based KB.

As long as the constraints are well defined and the knowledge base is well structured, a requirement from user the should be able to draw out an item that best fits the item the user wants.

### Case-Based
Similarity metrics are used to retrieve examples similar to the specified item. For this to work effectively, there are two crucial components: **similarity metrics** and **critiquing methods**.

*Note*: Not too much can be explained in detail in this section as each implementation of KB RS is so specific to each domain. It is very hard to generalize the best similarity metrics or critiquing methods; a lot of the decisions have to be made by the domain expert.

#### Similarity Metrics

With continuous variables, similarity can be something as simple as the difference between two numbers (e.g. the price of a car might be considered similar if the difference is close to 0) or it might be more complicated and consider more dimensions or use the standard deviation to set the similarity function.

With categorical data, the determination of similarity is much more challenging. Usually, domain hierarchies are used to measure the similarity within these variables. **Domain hierarchies** can be thought of as tree graphs where each category can be set to be under some category and those categories can also have sub-categories. The closer two objects are within the context of a domain hierarchy, the more similar it is. There are sometimes pre-made domain hierarchies (e.g. movie genre where commercial romance is closer to commercial comedy than an education documentary) but sometimes it has to be hand made by domain experts.

#### Critiquing

After a certain result is found using the similarity metrics and recommended to the user, the user has to be able to provide feedback and tweak the results to further match what they are looking for. This is also done in constraint-based KB recommenders. The user specifies change requests on one or more of the attributes of an item they like. The change request can be a directional critique (e.g. “smaller”) or a replacement critique (e.g. “different location”).

These can then be fed back into the similarity metric step to find one closer to what the user wants.

### Code Implementation
A lot of this is created with the help of another article. This article covers a simple implementation of a constraint-based KB recommender.

- *import data*

```python
import pandas as pd
import numpy as np
from ast import literal_eval

df = pd.read_csv("path/to/data")

# Select just relevant features
relevant_features = ['title','genres', 'release_date', 'runtime', 'vote_average', 'vote_count']
df = df[relevant_features]

# Print the dataframe
df.head()
```
<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-5-4.webp" loading="lazy" style="width: 700px;">
    </div>
</div>

- *preprocess data*

```python
#Convert release_date into pandas datetime format
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
# Extract year from release_date-column and store the values into a new year-column
df['year'] = pd.DatetimeIndex(df['release_date']).year
#Helper function to convert NaN to 0, if there are any, and all other years to integers.
def convert_int(x):
   try:
       return int(x)
   except:
       return 0
#Apply convert_int to the year feature
df['year'] = df['year'].apply(convert_int)
#Drop the release_date column
df = df.drop('release_date', axis=1)=
#Convert all NaN into stringified empty lists
df['genres'] = df['genres'].fillna('[]')
#Apply literal_eval to convert stringified empty lists to the list object
df['genres'] = df['genres'].apply(literal_eval)
#Convert list of dictionaries to a list of strings
df['genres'] = df['genres'].apply(lambda x: [i['name'].lower() for i in x] if isinstance(x, list) else [])

#Create a new feature by exploding genres
s = df.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)

#Name the new feature as 'genre'
s.name = 'genre'

#Create a new dataframe gen_df which by dropping the old 'genres' feature and adding the new 'genre'.
gen_df = df.drop('genres', axis=1).join(s)

#Print the head of the new gen_df
gen_df.head()
```

<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-5-5.webp" loading="lazy" style="width: 700px;">
    </div>
</div>

- *implementation of constraint-based recommender*
The filtering section is where the constraint satisfaction happens in its simplest form

```python
def build_chart(gen_df, percentile=0.8):
   #Ask for preferred genres
   print("Input preferred genre")
   genre = input()
   print(genre)
  
   #Ask for lower limit of duration
   print("Input shortest duration")
   low_time = int(input())
   print(low_time)
  
   #Ask for upper limit of duration
   print("Input longest duration")
   high_time = int(input())
   print(high_time)
  
   #Ask for lower limit of timeline
   print("Input earliest year")
   low_year = int(input())
   print(low_year)
  
   #Ask for upper limit of timeline
   print("Input latest year")
   high_year = int(input())
   print(high_year)
  
   #Define a new movies variable to store the preferred movies. Copy the contents of gen_df to movies
   movies = gen_df.copy()
  
   #Filter based on the condition
   movies = movies[(movies['genre'] == genre) &
                   (movies['runtime'] >= low_time) &
                   (movies['runtime'] <= high_time) &
                   (movies['year'] >= low_year) &
                   (movies['year'] <= high_year)]
  
   #Compute the values of C and m for the filtered movies
   C = movies['vote_average'].mean()
   m = movies['vote_count'].quantile(percentile)
  
   #Only consider movies that have higher than m votes. Save this in a new dataframe q_movies
   q_movies = movies.copy().loc[movies['vote_count'] >= m]
  
   #Calculate score using the IMDB formula
   q_movies['score'] = q_movies.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average'])
                                      + (m/(m+x['vote_count']) * C), axis=1)

   #Sort movies in descending order of their scores
   q_movies = q_movies.sort_values('score', ascending=False)
  
   return q_movies
```

After inputting [animation, 10, 40, 2000, and 2019], I get the following recommendations.
<div class="gallery-box">
    <div class="gallery">
        <img src="/images/rs-5-6.webp" loading="lazy" style="width: 700px;">
    </div>
</div>

### Resources
Aggarwal, C. C. (2016). Recommender systems: The textbook. SPRINGER.

Expert systems in Artificial Intelligence — Javatpoint. www.javatpoint.com. (n.d.). Retrieved February 26, 2023, from <a href="https://www.javatpoint.com/expert-systems-in-artificial-intelligence">https://www.javatpoint.com/expert-systems-in-artificial-intelligence</a>

Kantor, P. B., Rokach, L., Shapira, B., & Ricci, F. (2011). Recommender Systems Handbook. Springer.

Recommendation systems — knowledged-based recommender — Michael Fuchs Python. MFuchs. (2020, October 1). Retrieved February 26, 2023, from <a href="https://michael-fuchs-python.netlify.app/2020/10/01/recommendation-systems-knowledged-based-recommender/#build-the-knowledged-based-recommender">https://michael-fuchs-python.netlify.app/2020/10/01/recommendation-systems-knowledged-based-recommender/#build-the-knowledged-based-recommender</a>
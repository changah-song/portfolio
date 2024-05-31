---
layout: post
title:  Transformers for Dummies
description:
date:   2024-04-02 00:00:00 +0900
author: nolan
image:  '/images/transformer-thumbnail.avif'
tags:   [machine-learning, transformer]
tags_color: '#308000'
category: blog
---

If you've attempted to understand how Transformers work, you've probably come across this image from the groundbreaking paper - <a href="https://arxiv.org/pdf/1706.03762">Attention Is All You Need (Vaswanti et al., 2017).</a>

<div class="gallery-box">
    <div class="gallery">
        <img src="/images/t-1-1.png" loading="lazy" style="width: 700px;">
    </div>
    <p style="text-align: center; font-size: 15px; color: grey;">
        Transformer architecture (Vaswani et al., 2017)
    </p>
</div>


What are you looking at? Good question. I remember being completely lost when I saw this for the first and I spent a long time scouring resources on the internet trying understand what was going on behind all this.

*I'm assuming you, the reader, have some experience with Machine Learning and at least an understanding of Neural Networks (NNs). We will be using concepts such as embeddings, feed-forward, back-propagation, weights, linear layers, activation functions that are all covered when learning about NNs. *

*If not, Neural Networks for Dummies should be a good place to start.*

input → input embedding + positional embedding → copy paste 3 times → Query, Key, Value → Query and Key learns 

Basic idea is to train our model to learn how words relate to each other. In other words, how similar they are (in definition and context) from each other. If I input all of Shakespeare's works into the model, I want it to learn how all the words used in his work relate to each other. So I want it to be able to generate output that mimic his style. This is by knowing what words are often used together in a sentence (closer in context) and predicting what the next word of an input might be.

Imagine a 2-D plane with one axis being "size" and the other being "weight." We can put many things on this graph (as a point) to see how similar two things are in these two categories. For example, elephants and trucks should be closer together than mice and toasters. While this does a good job of showing which things are more similar in terms of size and weight, it doesn't capture other features such as color or being alive (in which case elephants and mice should be closer). To really capture every feature of everything, we need infinite dimensions (rather than 2-D). However, that is infeasible and so we just use hundreds of dimensions to map out everything in a hyperspace. We can't envision this as we are limited to 3-D but there can be a 500-dimensional space where every word in our English vocabulary is a point. The closer the points are, the closer the meaning and contextual use.

Great, so now we can talk about input embeddings. What we input (whether it is the Bible or other texts) turns into input embeddings. This means all words in the input gets a vector attached to it and the length of that vector is the embedding size. These vectors initially start off as random numbers and will be changed as we train the model. The longer the vector (embedding size), the more information, or linguistic feature, we can store about each word. The paper uses an embedding size of 512.
---
layout: post
title:  Front and Backward Propagation of Neural Networks
description:
date:   2024-04-01 00:00:00 +0900
author: nolan
image:  '/images/nn-thumbnail.avif'
tags:   [machine-learning, neural-network]
tags_color: '#308000'
category: blog
---

| I've come across some great resources that explain Neural Networks (mainly 3Blue1Brown for intuition & theory and this Medium article for code) and so this article is just to summarize what I've learned and is for my own learning and hopefully for others as well. 

This article assumes a basic understanding of what Neural Networks are and its basic components (layers, nodes, weights, and biases). I aim to explain clearly the theory behind forward and backpropagation and a clear code implementation behind it all.

I find the whole process to make more sense with two nodes.

### Forward Propagation
If you look at the following visualization, you'll see the we have all the basic components of a NN. Input node, output node, weights, biases, and activation functions (I will arbitrarily use the sigmoid activation function throughout this article). 
Then, we can express it in an equation. The weights and biases are initially just random numbers. The weights and biases are the only things that can affect the outcome of our NN. Given an input data, the model just does calculations until the final layer to give an output.
<div class="gallery-box">
    <div class="gallery">
        <img src="/images/nn-1-1.png" loading="lazy" style="width: 800px;">
    </div>
</div>
<div class="gallery-box">
    <div class="gallery">
        <img src="/images/nn-1-2.png" loading="lazy" style="width: 800px;">
    </div>
</div>

### Back Propagation
Now that we have an output, how do we train the network so that the output is more accurate? Right now the model just spits out a random output. To make it better, we have to tell the network that it did poorly and to make some changes to its previous layers, weights, and biases.
We can use an error value (I will use the Mean Squared Error for this article) to capture how different the output is from the true output. For our two-node example, you can see that the error between the output (Y) and the true value (red Y) is captured with E. 
<div class="gallery-box">
    <div class="gallery">
        <img src="/images/nn-1-3.png" loading="lazy" style="width: 800px;">
    </div>
</div>

You'll also notice that there are arrows pointing to the previous nodes, weights, and biases. This is our way of tweaking previous settings based on the error of our current state.
Think of E as the error of the entire network. How do we change it? Well, we can use partial derivatives to approach it. Partial derivatives, e.g. dY/dX, can be thought of as "how much of a change in X changes Y?" So basically, how much of the previous node (X), its weights (W), and biases (b) of the previous layer can I change to change our error (E)?
We want to minimize error. If you are familiar with gradients, you know that it always points in the direction of steepest ascent. What this means is that if we follow the gradient of a function, we end up going towards the maximum point. Since we want to go to the minimum point, we simply go the opposite direction.
W = W - dE/dW
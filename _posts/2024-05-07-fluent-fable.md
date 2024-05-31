---
layout: post
title: "Fluent Fable: Reading for Language Learning"
description: 
date:   2024-05-07 00:00:00 +0900
author: nolan
image:  '/images/ff-thumbnail.png'
tags:   [full-stack, react-native]
tags_color: '#AE3768'
category: project
---

**Introduction**

What’s the best way to learn a language? While there are many many approaches that have been developed throughout the years and many that I’ve personally tried, I’ve concluded that simply reading and engaging with material that you’re interested in is the best way to learn a new language and gain confidence in it. While the easiest way to expose yourself to a new language is to watch a show or series, I found it much too slow and unreliable.

Thus, I started reading. I started reading physical books first. It was nice but I had two main issues. 1) Regularly finding a book at your right language level in a genre you’re interested in is a struggle in itself. 2) Translating words and going from dictionary/translator to book back and forth disrupts the reading process. Then I started reading on my phone and found that issue 2 could be easily eliminated thanks to an easy way to find the definition of words as I was reading. However, issue 1 still remained and I also found it hard to be motivated finish an entire book; it was much too burdensome to stick with one book in an unfamiliar language

Well, if you can’t discipline yourself, create something that lowers the need for said discipline right?

So, I started creating a simple mobile app using React Native as I knew some Javascript. I wanted to make the app very simple. One where you could enter, read a small passage that matched my level, translate any unfamiliar words, and exit. It would also save unfamiliar words and allow for the use of flashcards to memorize the words if you so desired.

<div class="gallery-box">
    <div class="gallery">
        <img src="/images/fluent-fable-viz.png" loading="lazy" style="width: 750px;">
    </div>
</div>

<div style="display: flex; gap: 10px;">
  <div style="position:relative;width:33.33%;padding-top:56.25%;">
    <iframe src="https://player.vimeo.com/video/952394445?badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479" frameborder="0" allow="autoplay; fullscreen; picture-in-picture; clipboard-write" style="position:absolute;top:0;left:0;width:100%;height:100%;" title="first"></iframe>
  </div>
  
  <div style="position:relative;width:33.33%;padding-top:56.25%;">
    <iframe src="https://player.vimeo.com/video/952394472?badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479" frameborder="0" allow="autoplay; fullscreen; picture-in-picture; clipboard-write" style="position:absolute;top:0;left:0;width:100%;height:100%;" title="second"></iframe>
  </div>
  
  <div style="position:relative;width:33.33%;padding-top:56.25%;">
    <iframe src="https://player.vimeo.com/video/952395102?badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479" frameborder="0" allow="autoplay; fullscreen; picture-in-picture; clipboard-write" style="position:absolute;top:0;left:0;width:100%;height:100%;" title="third"></iframe>
  </div>
</div>
<script src="https://player.vimeo.com/api/player.js"></script>  

---

개인적으로 언어를 배우면서 꼭 존재해야 되겠다고 생각한 후 시작한 프로젝트로, React Native 대해서 하나도 몰랐던 제가 많은 걸 배우면서 재밌게 만든 앱입니다.

저 처럼 한국 국민인데 외국에서 삶 대부분을 살아오셨다면 이 앱의 목적과 기능들이 이해가 더 잘 될 거에요. 

한글은 어느 정도 이해하고 말할 수 있지만, 어휘에서 걸리는 절망적인 느낌.. 한국어가 영어를 모국어로 삼는 사람들에게는 가장 어려운 언어라는 사실과 한국어의 단어 수량이 엄청나다는 게 그 느낌을 설명하겠죠.

제가 한국어뿐만 아니라 중국어도 열심히 공부하면서 깨달은 점은 책만 많이 읽으면 언어는 따라오게 된있다라는 거에요. 그래서 책을 읽을 때 가장 좋은 경험을 줄 수 있는 앱을 만들기로했어요. 

**Features, 기능**

그래서, 이 앱을 많은 API를 통해서 다음의 기능을 담은 eReader을 개발했어요:

1. 읽고 싶은 책 고르기 (Reader API)
2. 단어 선택 후 사전 검색 (Korea Dictionary API)
3. 단어 선택 후 어근 필터 (FastAPI를 사용해서 KONLPY API)
4. 단어 선택 후 번역기 검색 (Translator API)
5. 단어 저장 후 플래시 카드로 복습 (SQLite)

**What I Learned, 배운 것**

1. Asynchronous functions
2. COR, FastAPI local server
3. Object-Oriented Programming 및 Separation of Concerns 
4. React Hooks (State, Effect, Context)
5. Animation (PanResponder and Animated)
6. Styling and UI

Documentation of Progress: <a href="https://github.com/paul-song-minerva/fluent-fable/tree/main/frontend">https://github.com/paul-song-minerva/fluent-fable/tree/main/frontend</a>
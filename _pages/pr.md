---
layout: archive
title: "Projects"
permalink: /projects/
author_profile: true
classes: wide
header :
    image: "https://maelfabien.github.io/assets/images/wolf.jpg"
---


### Big Data : Design and implementation of high-performance resilient storage system on AWS to analyse GDELT Database

The GDELT Project monitors the world's broadcast, print, and web news from nearly every corner of every country in over 100 languages and identifies the people, locations, organizations, themes, sources, emotions, counts, quotes ...in the entire world. With new files uploaded every 15 minutes, GDELT data bases contain more than 500 Gb of zipped data for the single year 2018.

We designed a high-performance distributed storage system on AWS that can analyze the events of the year 2018 through their story in the world media collected by GDELT. The goal is to analyze trends and relationships between different country actors.
* We used spark as an ETL , with its native language Scala: we created a first script that loads the data in S3 , format parquet, and a second script scala to do the intermediate data processing and deposit the cleaned data on MongoDB & Cassandra instances deployed on AWS.
* We request the databases from python with the appropriate connectors of each base (with pymongo for MongoDB for example)
* We used Flask for the visualization part

![image](https://raw.githubusercontent.com/kasamoh/NoSQL/master/Projet_gdelt/Screenshot_gdelt.png)

Keywords: AWS EC2, ZooKeeper, S3, Zepplin, scala, mongoDB, Cassandra, Flask, Python

See GitHub page : <span style="color:blue">[https://github.com/kasamoh/NoSQL/tree/master/Projet_gdelt](https://github.com/kasamoh/NoSQL/tree/master/Projet_gdelt)</span>


<br/>
<br/>
<br/>
<br/>
<br/>



### Reinforcement Learning : Deep Reinforcement learning for recommender system

Build A recommender system for Vente-privee.com using Reinforcement Learning and Bayesian Networks . 
Keys : Contextual bandits , Gym , Neural Networks , Bayesian Analysis , Python , Docker , Tensorflow


![image](https://raw.githubusercontent.com/mohameddhaoui/mohameddhaoui.github.io/master/images/vepee.JPG)


See GitHub page : <span style="color:blue">[https://github.com/kasamoh/vente-privee_telecom_reco_gym](https://github.com/kasamoh/vente-privee_telecom_reco_gym)</span>

<br/>
<br/>
<br/>
<br/>
<br/>



### Metric Learning and XGBOOST : Estimating a position from a received signal strength for IoT sensors

Smart devices such as IoT sensors use low energy consuming networks such as the ones provided by Sigfox or Lora. But without using GPS networks, it becomes harder to estimate the position of the sensor. The aim of this study is to provide a geolocation estimation using Received Signal Strength Indicator in the context of IoT. The aim is to allow a geolocation of lowconsumption connected devices using the Sigfox network. State of the art modelsare able to be precise to the nearest kilometer in urban areas, and around tenkilometers in less populated areas. 

Keys : Metric Learning , KNN , Xgboost , IoT , Python

![image](https://www.simultrans.com/hs-fs/hubfs/Challenges%20of%20Localizing%20IoT%20content.png?width=560&height=315&name=Challenges%20of%20Localizing%20IoT%20content.png)

See GitHub page: <span style="color:blue">[https://github.com/kasamoh/IoT](https://github.com/kasamoh/IoT)</span>

<br/>
<br/>
<br/>
<br/>
<br/>

### Flask : an app for movies website 
A design of a Flask app for movies website with python and MongoDB


![image](https://raw.githubusercontent.com/kasamoh/NoSQL/master/mflix.png)

keywords :  Flask ,Python, MongoDB , HTML , CSS , Boostrap 
See GitHub page : <span style="color:blue">[https://github.com/kasamoh/NoSQL/tree/master/MongoDB/mflix](https://github.com/kasamoh/NoSQL/tree/master/MongoDB/mflix)</span>

<br/>
<br/>
<br/>
<br/>
<br/>

### Visualization : Interactive Map of France using D3js

In this project, I created an interactive map with a few main features :

- Display the map of France (population and density)
- Change the color of the map with a simple button
- Display a tooltip with the name of the city and the postal code
- Display density and population histograms

First design : 

![](https://raw.githubusercontent.com/kasamoh/Data-analysis/master/Data%20Visualization/D3js/France/images/d3js_res.png)



Second design : 
![](https://raw.githubusercontent.com/kasamoh/Data-analysis/master/Data%20Visualization/D3js/France/images/Capture.JPG)

keywords :  Flask ,Python, D3js , javascript , HTML , CSS , Boostrap 
See GitHub page : <span style="color:blue">[https://github.com/kasamoh/Data-analysis/tree/master/Data%20Visualization/D3js](https://github.com/kasamoh/Data-analysis/tree/master/Data%20Visualization/D3js)</span>

<br/>
<br/>
<br/>
<br/>
<br/>

## Web : Wikipedia Web App


According to ReadWriteWeb, all articles in the English version of Wikipedia lead to the article “Philosophy”. If you click on the first link of each article, you will come across the Philosophy article after a while.

In order to verify this assertion, I developped a small WebApp with 2 simple options :

- A manual Wikipedia Link explorer that requests Wikipedia’s API
- An automatic exploration that systematically chooses the first link among results


![](https://raw.githubusercontent.com/kasamoh/WebApp/master/WikiAPP/static/wikipedia.JPG)

See GitHub page : <span style="color:blue">[https://github.com/kasamoh/WebApp/tree/master/WikiAPP](https://github.com/kasamoh/WebApp/tree/master/WikiAPP)</span>




<br/>
<br/>
<br/>
<br/>
<br/>

### Classification  :Click Through Rate Prediction using AVAZU dataset

Many ads are actually sold on a "pay-per-click" (PPC) basis, meaning the company only pays for ad clicks, not ad views. Thus your optimal approach (as a search engine) is actually to choose an ad based on "expected value", meaning the price of a click times the likelihood that the ad will be clicked. In other words, a $1.00 ad with a 5% probability of being clicked has an expected value of $0.05, whereas a $2.00 ad with a 1% probability of being clicked has an expected value of only $0.02. In this case, you would choose to display the first ad.

In order for you to maximize expected value, you therefore need to accurately predict the likelihood that a given ad will be clicked, also known as "click-through rate" (CTR).

![]() https://storage.googleapis.com/kaggle-competitions/kaggle/4002/media/banner_ctr.jpg

See GitHub page: <span style="color:blue">[https://github.com/kasamoh/Big-Data-mining/tree/master/AVAZU_CTR_Analysis](https://github.com/kasamoh/Big-Data-mining/tree/master/AVAZU_CTR_Analysis)</span>


<br/>
<br/>
<br/>
<br/>
<br/>


### Classification : Predicting the predominant kind of tree (Kaggle)

In this challenge , I am trying to predict the forest cover type (the predominant kind of tree cover) from strictly cartographic variables (as opposed to remotely sensed data) . 

See GitHub page : <span style="color:blue">[https://github.com/kasamoh/Data-analysis/tree/master/Kaggle](https://github.com/kasamoh/Data-analysis/tree/master/Kaggle)</span>


<br/>
<br/>
<br/>
<br/>
<br/>

### Cyber Security (Splunk)

I used Splunk in a Cyber Security Project. The aim of the project was to identify the source of a data leakage within the company. We went through the logs, identified suspect IP addresses, found the source of the attack (a corrupted PDF), estimated the volume of data stolen, and proposed immediate actions. We detailed the Diamond Model, the Cyber Kill Chain, and developped general perspectives for the Cyber Threat Intelligence of the company.

<embed src="https://github.com/mohameddhaoui/mohameddhaoui.github.io/raw/master/assets/images/INF726_Cybersecu_TP_Splunk_Dhaoui_Reynal_Soufflet.pdf" type="application/pdf" width="600px" height="500px" />


<br/>
<br/>
<br/>
<br/>
<br/>
### Scala mini-projects : 

- Financing rate prediction of Kickstarter.com projects : https://github.com/kasamoh/Big-Data-mining/tree/master/Scala/KItStarterProject
- Implementation a distributed k-means algorithm which clusters posts on the popular question-answer platform StackOverflow according to their score : https://github.com/kasamoh/Big-Data-mining/tree/master/Scala/StackOverflowProject
- Using full-text data from Wikipedia to produce a rudimentary metric of how popular a programming language is : https://github.com/kasamoh/Big-Data-mining/tree/master/Scala/Wikipediaproject

<br/>
<br/>
<br/>
<br/>
<br/>
### Deep Learning : Image Processing and learning  

Below some image 
- Realtime Landmarks recognition using Transfer Learning and open CV 
![](https://github.com/kasamoh/Image_processing_learning/Landmarks_Detection/Screenshot%20from%202019-04-04%2010-52-00.png)

- Gender detection  using deep learning with keras and cvlib
![](https://github.com/kasamoh/Image_processing_learning/Gender_detection/woman_men.png)

- Age Prediction in Keras 

<br/>
<br/>
<br/>
<br/>
<br/>
## Hackathons

[Predicting the Song of the year ](https://github.com/kasamoh/Hackathon/blob/master/Hackaton_report.ipynb)

[Predicting the Song of the year (2/3)](https://maelfabien.github.io/Hack-2/)

[Predicting the Song of the year (3/3)](https://maelfabien.github.io/Hack-3/)

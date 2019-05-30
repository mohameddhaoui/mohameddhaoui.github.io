---
layout: archive
title: "Projects"
permalink: /projects/
author_profile: true
classes: wide
header :
    image: "https://maelfabien.github.io/assets/images/wolf.jpg"
---


## GitHub Projects

### Big Data : Design and implementation of high-performance resilient storage system on AWS to analyse GDELT Database

The GDELT Project monitors the world's broadcast, print, and web news from nearly every corner of every country in over 100 languages and identifies the people, locations, organizations, themes, sources, emotions, counts, quotes ...in the entire world. With new files uploaded every 15 minutes, GDELT data bases contain more than 500 Gb of zipped data for the single year 2018.

We designed a high-performance distributed storage system on AWS that can analyze the events of the year 2018 through their story in the world media collected by GDELT. The goal is to analyze trends and relationships between different country actors.
* We used spark as an ETL , with its native language Scala: we created a first script that loads the data in S3 , format parquet, and a second script scala to do the intermediate data processing and deposit the cleaned data on MongoDB & Cassandra instances deployed on AWS.
* We request the databases from python with the appropriate connectors of each base (with pymongo for MongoDB for example)
* We used Flask for the visualization part

![image](https://raw.githubusercontent.com/kasamoh/NoSQL/master/Projet_gdelt/Screenshot_gdelt.png)

Keywords: AWS EC2, ZooKeeper, S3, Zepplin, scala, mongoDB, Cassandra, Flask, Python

See GitHub page : <span style="color:blue">[https://github.com/kasamoh/NoSQL/tree/master/Projet_gdelt](https://github.com/kasamoh/NoSQL/tree/master/Projet_gdelt)</span>



### Flask : an app for movies website 
A design of a Flask app for movies website with python and MongoDB


![image](https://github.com/kasamoh/NoSQL/blob/master/mflix.png)

keywords :  Flask ,Python, MongoDB , HTML , CSS , Boostrap 
See GitHub page : <span style="color:blue">[https://github.com/kasamoh/NoSQL/tree/master/MongoDB/mflix](https://github.com/kasamoh/NoSQL/tree/master/MongoDB/mflix)</span>


### Visualization : Interactive Map of France using D3js

In this project, I created an interactive map with a few main features :

-Display the map of France (population and density)
-Change the color of the map with a simple button
-Display a tooltip with the name of the city and the postal code
-Display density and population histograms


![La France](https://raw.githubusercontent.com/kasamoh/Data-analysis/master/Data%20Visualization/D3js/France/images/par1.png)

![](https://raw.githubusercontent.com/kasamoh/Data-analysis/master/Data%20Visualization/D3js/France/images/d3js_res.png)

![](https://raw.githubusercontent.com/kasamoh/Data-analysis/master/Data%20Visualization/D3js/France/images/Capture.JPG)

keywords :  Flask ,Python, D3js , javascript , HTML , CSS , Boostrap 
See GitHub page : <span style="color:blue">[https://github.com/kasamoh/Data-analysis/tree/master/Data%20Visualization/D3js](https://github.com/kasamoh/Data-analysis/tree/master/Data%20Visualization/D3js)</span>




### Reinforcement Learning : Deep Reinforcement learning for recommender system

Build A recommender system for Vente-privee.com using Reinforcement Learning and Bayesian Networks . 
Keys : Contextual bandits , Gym , Neural Networks , Bayesian Analysis , Python , Docker , Tensorflow


![image](https://raw.githubusercontent.com/mohameddhaoui/mohameddhaoui.github.io/master/images/vepee.JPG)
See GitHub page : <span style="color:blue">[https://github.com/kasamoh/vente-privee_telecom_reco_gym](https://github.com/kasamoh/vente-privee_telecom_reco_gym)</span>



### Deep Learning : Multimodal Sentiment Analysis (Text, Sound, Video)

In this project, I am exploring state of the art models in multimodal sentiment analysis. We have chosen to explore textual, sound and video inputs and develop an ensemble model that gathers the information from all these sources and displays it in a clear and interpretable way.

I am currently working on a Tensorflow.js implementation of this project. Don't hesite to Star the project if you like it.

See GitHub page : <span style="color:blue">[https://github.com/maelfabien/Mutlimodal-Sentiment-Analysis](https://github.com/maelfabien/Mutlimodal-Sentiment-Analysis)</span>

<embed src="https://maelfabien.github.io/assets/images/PE.pdf" type="application/pdf" width="600px" height="500px" />

### Metric Learning and XGBOOST : Estimating a position from a received signal strength for IoT sensors

Smart devices such as IoT sensors use low energy consuming networks such as the ones provided by Sigfox or Lora. But without using GPS networks, it becomes harder to estimate the position of the sensor. The aim of this study is to provide a geolocation estimation using Received Signal Strength Indicator in the context of IoT. The aim is to allow a geolocation of lowconsumption connected devices using the Sigfox network. State of the art modelsare able to be precise to the nearest kilometer in urban areas, and around tenkilometers in less populated areas. 

Keys : Metric Learning , KNN , Xgboost , IoT , Python

![image](https://www.simultrans.com/hs-fs/hubfs/Challenges%20of%20Localizing%20IoT%20content.png?width=560&height=315&name=Challenges%20of%20Localizing%20IoT%20content.png)

See GitHub page: <span style="color:blue">[https://github.com/kasamoh/IoT](https://github.com/kasamoh/IoT)</span>




<embed src="https://maelfabien.github.io/assets/images/RSSI.pdf" type="application/pdf" width="600px" height="500px" />

### NLP : Analyzing GitHub Pull Requests

In this project, I have been looking at comments of developers on GitHub pull requests in order to :
- determine the main topics (LSA Topic Modelling)
- identify clusters of words (KMeans)
- predict if a merge will occur after the comment (Bag Of Words, TF-IDF)
- predict the time before the merge

See GitHub page: <span style="color:blue">[https://github.com/maelfabien/Analyze-Github-Pull-Requests](https://github.com/maelfabien/Analyze-Github-Pull-Requests)</span>

<embed src="https://maelfabien.github.io/assets/images/NLP.pdf" type="application/pdf" width="600px" height="500px" />

### Classification : Predicting the predominant kind of tree (Kaggle)

In this challenge , I am trying to predict the forest cover type (the predominant kind of tree cover) from strictly cartographic variables (as opposed to remotely sensed data) . 

See GitHub page : <span style="color:blue">[https://github.com/kasamoh/Data-analysis/tree/master/Kaggle](https://github.com/kasamoh/Data-analysis/tree/master/Kaggle)</span>

### Cyber Security (Splunk)

I used Splunk in a Cyber Security Project. The aim of the project was to identify the source of a data leakage within the company. We went through the logs, identified suspect IP addresses, found the source of the attack (a corrupted PDF), estimated the volume of data stolen, and proposed immediate actions. We detailed the Diamond Model, the Cyber Kill Chain, and developped general perspectives for the Cyber Threat Intelligence of the company.

<embed src="https://github.com/mohameddhaoui/mohameddhaoui.github.io/blob/master/assets/images/INF726_Cybersecu_TP_Splunk_Dhaoui_Reynal_Soufflet.pdf" type="application/pdf" width="600px" height="500px" />


## Hackathons

[Predicting the Song of the year (1/3)](https://maelfabien.github.io/Hack-1/)

[Predicting the Song of the year (2/3)](https://maelfabien.github.io/Hack-2/)

[Predicting the Song of the year (3/3)](https://maelfabien.github.io/Hack-3/)

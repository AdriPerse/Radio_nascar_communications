# NASCAR Radio Communication Classification

This repository contains the code for a Streamlit application that allows users to test our natural language processing (NLP) models for classifying radio communications in a car racing context. 

This project was conducted as a capstone project under the supervision of [Yeow Teck Keat](https://www.linkedin.com/in/yeow-tk/) from [Rolos](https://rolos.com/) during our Bootcamp at [Constructor](https://learning.constructor.org/). The team, comprised of [Adriano Persegani Daguzan](https://www.linkedin.com/in/adriano-persegani/), [Tuyen Nguyen Thi](https://www.linkedin.com/in/tuyen-nguyen-thi-7576967b/), [Yeeun Kim](https://www.linkedin.com/in/yeeun-kim-bba19b15b/), and [Ibrahima Ba](https://www.linkedin.com/in/ibrahima-ba-data-scientist-germany/), collaborated to help NASCAR, a car racing company, classify the information being transmitted during a race for faster and more accurate responses. 

<img src="data/images/streamlit-demo.gif" width="800"/>

## Problem Statement

During a car race, a lot of information is communicated between drivers and engineers. Among this information, some might not be crucial while others, like messages related to tire fuel or emergency situations, should never be missed. The challenge is to accurately and quickly deliver only the important information to the pilot team. 

Our solution to this problem was to use NLP techniques to automatically and accurately classify the radio communications. We worked with data already transformed from voice radio communication into text. 

## Our Approach

1. **Unsupervised Machine Learning:** We used unsupervised machine learning models to identify the topics based on word distribution. This approach allowed us to group similar topics together.

2. **Supervised Machine Learning:** To enhance the accuracy of the classification, we employed supervised machine learning models. We trained the models on 600 messages with 29 topics with the aim to accurately classify the topic of new, unseen messages. 

## Models Used

We experimented with different types of models including:

- **Rule-Based Model:** This is a simple, classical machine learning model. It, however, did not perform as well as the other models, possibly due to its simple algorithm and its performance on smaller datasets. 

- **Transformer Models:** These models use more advanced technology that can analyze the context of a message. They performed much better than the rule-based model. 

- **Few-shot Learning Models:** These models performed the best, even with very few training data. They were able to predict with more than 50% accuracy on some topics like fuel or tires.

## Future Work

Although our model is already up and running, we believe that with more computational power, more time, and more data, we could improve its accuracy. We could consider more context or even whole conversations for future improvements.

## Try It Out

You can test our models yourself on our web app at [radionascar.com](http://www.radionascar.com). We have uploaded two models: a simple one and a transformer one which is similar technology to Chat GPT. 

Your feedback would be much appreciated. 

## Connect With Us

Feel free to connect with us on LinkedIn. 

- [Adriano Persegani Daguzan](https://www.linkedin.com/in/adriano-persegani/)
- [Tuyen Nguyen Thi](https://www.linkedin.com/in/tuyen-nguyen-thi-7576967b/)
- [Yeeun Kim](https://www.linkedin.com/in/yeeun-kim-bba19b15b/)
- [Ibrahima Ba](https://www.linkedin.com/in/ibrahima-ba-data-scientist-germany/)

Thank you very much for your time.

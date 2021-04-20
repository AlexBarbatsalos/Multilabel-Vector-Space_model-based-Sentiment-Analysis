# Multilabel-Vector-Space_model-based-Sentiment-Analysis

## General Information 

This project consists of code enabling students, researchers and people interested in text mining to implement a Sentiment-Analysis(SA) framework based on 
Vector Space Models (see Mikolov et al. 2013).

In the follwowing I will list some distinctive features of our approach for better user orientation.

(1) The SA framework is based on SentiArt, a VSM-based SA tool, for reference see the important papers

(2) Our objective is finding sets of labels for DISCRETE EMOTIONS in vector space to reliably predict human-rated sentiment across two GERMAN data sets:
      - The DENN_BAWL ('Discrete Emotion Norms for Nouns: Berlin Affective Word List';see Briesemeister et al., 2011)
      - A Harry Potter chapter consisting of 120 Sentences
(3) In order to achieve that, we sample words, simmilar to five discrete emotion labels (ANGST, FREUDE, WUT, TRAUER, EKEL) from vector space and evaluate the sample with respect to predictive strength of the sampled units
(4) We use the resulting sets of labels per discrete emotions in (1) a simple Correlation and (2) a Classification with the human ratings in the datasets.


## Folder Content

## Usage Tips

### important papers

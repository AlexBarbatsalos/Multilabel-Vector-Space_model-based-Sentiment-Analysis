# Multilabel-Vector-Space_model-based-Sentiment-Analysis

## General Information 

This project consists of code enabling students, researchers and people interested in text mining to implement a Sentiment-Analysis(SA) framework based on 
Vector Space Models (see Mikolov et al. 2013).

In the follwowing I will list some distinctive features of our approach for better user orientation.

(1) The SA framework is based on SentiArt, a VSM-based SA tool, for reference see the important papers

(2) Our objective is finding sets of labels for DISCRETE EMOTIONS in vector space to reliably predict human-rated sentiment across two GERMAN data sets:
      
   -The DENN_BAWL ('Discrete Emotion Norms for Nouns: Berlin Affective Word List';see Briesemeister et al., 2011)
   
   -A Harry Potter chapter consisting of 120 Sentences
      
(3) In order to achieve that, we sample words, simmilar to five discrete emotion labels (ANGST, FREUDE, WUT, TRAUER, EKEL) from vector space and evaluate the sample with respect to predictive strength of the sampled units

(4) We use the resulting sets of labels per discrete emotions in (1) a simple Correlation and (2) a Classification with the human ratings in the datasets.


## Folder Content

#### data analysis code

(1) SentiArt.py: A Python class for the setup of a SentiArt-table

(2) DENN_BAWL Validation.py: Code for evaluating the sampled multilabel sets from vector space in a validation process based on the DENN_BAWL. This code also runs the correlation and classification tasks on the dataset

(3) harry_potter_data_preprocessing.py: data for preprocessing of harry potter sentence data. based on the output of a treetaggerwrapper (see data folder)

(4) harry_potter_analysis.py: runs correlation and classification tasks on the dataset


#### data
This folder is incomplete up to now, new content will be added step by step

(1) harry_preprocessed.xlsx: excel-file containing relevant preprocessing output for the harry potter dataset for subsequent analysis

(2) treetagger_harry_output.txt: output of the TreeTagger for Windows (https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/)

(3) tsne frames: those frames are the result of a tsne-based reduction of the full 300-dim VSM, you can either compute them yourself (see DENN_BAWL Validation.py), or   directly load them with pd.read_excel

For any further data  (DENN_BAWL, Harry Potter raw data) please contact me (alex-ts@gmx.de).

#### charts/graphs
currently empty

The folder will include code for graphics, which support the understanding and/or presentation of the results of the analysis.


### important papers
Briesemeister, B. B., Kuchinke, L., Jacbos, A. M. (2011). Discrete emotion norms for nouns: Berlin affective word list (DENN–BAWL). Behav Res, 43, 441-448.

Jacobs, A. M., & Kinder, A.(2019). Computing the Affective-Aesthetic Potential of Literary Texts. AI 2020, 1, 11-27.

Mikolov, T., Chen, K., Corrado, G., Dean, J.(2013). Efficient Estimation of Word Representations in Vector Space. Proceedings of ICLR Workshops Track, 1-12.




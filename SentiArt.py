import os
import gensim
import nltk
import sklearn
import numpy as np
import pandas as pd


class SentiArt_Manager:

    description = '''Author: Alexander Barbatsalos

This class is a tool for setting up and validating a SentiArt-table based on a given VSM.

For initialization only (1)a table of words for which cosine-similarities should be computed and (2)a list of lists of labels is required.

The VSM is initialized via the self.load_model() function.
The lables are preprocessed for operating in the self.manage_labels() function, yielding a dict of labels as keys with sublabels as values.

The self.set_up_table() function computes the values of interest and represents them as a pd.DataFrame, which can be saved using the self.save_as_file() function.

For Validation use the self.cross_validate_table() function.'''.replace('\n', ' ')

    dependencies = '''
For using this class, the following libraries have to be installed:

os
gensim
nltk
sklearn
numpy
pandas'''.replace('\n',' ')

    def __init__(self, table_words, labels):
        '''
        initializes the Manager.
        here all properties of the Manager are listed.
        some can be directly initialized, some have to be
        included in the Model by the methods
        '''
        self.table_words = table_words
        self.labels = labels
        self.model = None
        self.SentiArt_table = None


    def load_model(self, model):
        '''
        loads VSM model with gensim library and converts it to vectorized format
        '''
        
        from gensim.models import KeyedVectors

        vsm = KeyedVectors.load_word2vec_format(model)

        try:
            vectorized = vsm.wv
            self.model = vectorized

        except:
            print('No conversion possible or model already in vectorized form.')
            self.model = vsm
         
        
        return print('successful import: ' + model)
    
    
    def align_table_words(self):
        '''
        this function excludes all words from self.table_words, which are not in the model, and in this way
        avoiding errors or rule out their influence on follow-up computations'''
        model_vocab = [key for key in list(self.model.vocab.keys())]
        capitalized_words = 0
        
        for word in self.table_words:
            if word not in model_vocab:
                word = word.capitalize()
                capitalized_words += 1
            
            
        
       
        return print('Number of capitalized words: {no}'.format(no=capitalized_words))
        



    def set_up_table(self, accumulation_scheme = [0]):
        '''
        returns Sentiart-table in the form of a pandas DataFrame
        input:
            model in the form of self.model (see self.load_model)
            table_words in the form of a list
            label list(see manage_labels)
            accumulation_scheme is a scheme according to which single- or multilabel-computations done
                default: [1] means that per label-key only one label is used i.e. single-label-computation
                example for multilabel:
                            [0,4,9] --> one column only using the first label, one column using the mean of the first 5 labels,
                                         one column using the mean of first 10 labels of the labels in self.label_list.

        output:
            SentiArt-table with standardized cosine_similarities between labels and words of interest as pd.DataFrame
        '''
        
        
        
        
        column_setup = []

        for label in self.labels.keys():
            for i in range(len(accumulation_scheme)):
                column_setup.append(str(label) + ' ' + str(accumulation_scheme[i]+1))
        
        table = pd.DataFrame(index = self.table_words, columns = column_setup)
        
        ### filling empty table with cosine similarities
        
        for word in table.index:
            for label in list(table.columns):
                try:
                    table.loc[word][label] = np.mean([self.model.similarity(word,sublabel) for sublabel in self.labels[label.split()[0]][:int(label.split()[1])]])
                except:
                    table.loc[word][label] = np.mean([self.model.similarity(word.capitalize(),sublabel) for sublabel in self.labels[label.split()[0]][:int(label.split()[1])]])
                    
        
        self.SentiArt_table = table
        
        return table
        
        
    def standardize_values(self):

        # Means and standard deviations per column mean is dict[key][0] and std is dict[key][1]

        means0_stds1 = dict()

        for label in list(self.SentiArt_table.columns):
            means0_stds1[label] = [np.mean(self.SentiArt_table[label]), np.std(self.SentiArt_table[label])]
            
            
        for word in self.SentiArt_table.index:
            for label in list(self.SentiArt_table.columns):
                if self.SentiArt_table.loc[word][label] == 0:
                    continue
                else: 
                    self.SentiArt_table.loc[word][label] = (self.SentiArt_table.loc[word][label] - means0_stds1[label][0])/means0_stds1[label][1]
        
        
        return self.SentiArt_table



    def save_as_file(self, path, sheet_name = 'sheet 1', save_format='excel'):
        '''
        saves DataFrame as .csv or .xlsx, (default is 'excel', otherwise 'csv')
        '''
        if save_format == 'excel':
            self.SentiArt_table.to_excel(path, sheet_name = sheet_name)

        elif save_format == 'csv':
            self.SentiArt_table.to_csv(path_or_buf=path)

        
        return print('Successfully saved to {}, named {}'.format(path, sheet_name))
    
    def compute_single_column(self, words, label_dict):
        '''
        computes just a single column of a sentiArt table
        faster and more flexible computation possible for exploratory sentiment analysis
        second input is a dict with a single key, which is the label and a list of corresponding sublabels as values
        output is a pd.Series'''
        
        label_col = pd.Series(index=words,dtype='float64')
        non_stand_list = []
        
      
        for word in label_col.index:
            try:
                non_standardized = np.mean([self.model.similarity(word,sublabel) for sublabel in label_dict[list(label_dict.keys())[0]]])
            except: 
                non_standardized = np.mean([self.model.similarity(word.capitalize(),sublabel) for sublabel in label_dict[list(label_dict.keys())[0]]])
            
            non_stand_list.append(non_standardized)
            
        label_mean = np.mean(non_stand_list)
        label_std = np.std(non_stand_list)
        
        for i in range(len(label_col.index)):
            label_col.iloc[i] = (non_stand_list[i] - label_mean)/label_std
            
        info = list(label_dict.keys())[0], str(len(label_dict[list(label_dict.keys())[0]]))
            
        return info, label_col




    

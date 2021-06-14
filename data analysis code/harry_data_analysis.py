###############   HARRY POTTER SENTENCE DATA ANALYSIS ##############

import os
import pandas as pd
import numpy as np
import nltk
import gensim
import sklearn
import matplotlib.pyplot as plt

wd=''       ### INSERT WORKING DIRECTORY HERE ###
os.chdir(wd)

import SentiArt   #has to be in working directory



'''
import of necessary data:
- preprocessed harry potter data (see 'harry_potter_data_preprocessing.py')
- multilabel dictionaries for the 2 discrete emotions of interest
    (tsne and 300-dim model)
'''

data_path = ''
harry_data = pd.read_excel(data_path, engine='openpyxl').drop('Unnamed: 0', axis='columns')

for i in range(len(harry_data['nouns'])):
    harry_data['nouns'][i] = harry_data['nouns'][i].\
    replace('[','').replace(']','').replace('\'','').replace(' ','').split(',')


word_table = [word for no in harry_data['nouns'].values for word in no]



angst_dict_full = {'Angst': ['Angst', 'Verlassenheitsängsten', 
                        'Bedrohungsgefühlen', 'Verfolgungsängste', 
                        'Bedrohungsangst', 'Unsicherheitsgefühl', 
                        'Angstgefühlen', 'Ohnmachtsgefühle', 
                        'Panikgefühle', 'Todesängste', 
                        'Hilflosigkeitsgefühle', 
                        'Angstmache', 'Ängstlichkeit', 
                        'Angstschreien', 'Schamgefühle']}

angst_dict_tsne = {'Angst': ['Angst', 'Gewissensangst', 
                        'Ohnmachtsgefühlen', 'Hexenangst', 
                        'Angstquelle', 'Ohnmachtsangst', 
                        'Verlorenheitsgefühle', 'Angstlust', 
                        'Bedrohungsgefühle', 'verängstigen', 
                        'bedrängst', 'Zukunftsangst', 
                        'Redeangst', 'Schwellenangst', 
                        'Angstvoll']}



freude_dict_full = {'Freude': ['Freude', 'Sommerfreuden', 
                          'Sinnesfreude', 'Entdeckerfreude', 
                          'freuden', 'freudestrahlend', 
                          'Begeisterung', 'freudevollen', 
                          'Lebensfreuden', 'freudetrunken', 
                          'Vorfreude', 'freudvolle', 
                          'freudige', 'Glückseeligkeit', 
                          'Wohlgefühle']}


freude_dict_tsne = {'Freude': ['Freude', 'Sommerfreude', 
                          'Kinderfreude', 'begeisterung', 
                          'Trinkfreude', 'Vorfreude', 
                          'Sinnenfreuden', 'Begeisterte', 
                          'Fröhlichkeit', 'Begeisterungsrufe', 
                          'Erwartungsfreude', 'Kauffreude', 
                          'frohem', 'Sangesfreude', 
                          'Soap-Begeisterung']}


## import vsm with SentiArtManager
senti = SentiArt.SentiArt_Manager(word_table, None)
model=''
senti.load_model(model)

# check for full vocab-overlap
for word in word_table:
    if word not in senti.model.vocab:
        print(word)

## Compute normalized cosine similarities of word_set and multilabel-sets and
## arrange them sentence-wise

angst_single = senti.compute_single_column(word_table,{'Angst':['Angst']})
angst_multi_full = senti.compute_single_column(word_table,angst_dict_full)
angst_multi_tsne = senti.compute_single_column(word_table,angst_dict_tsne)

freude_single = senti.compute_single_column(word_table,{'Freude':['Freude']})
freude_multi_full = senti.compute_single_column(word_table,freude_dict_full)
freude_multi_tsne = senti.compute_single_column(word_table,freude_dict_tsne)


angst_single_per_sent = dict((k,list()) for k in range(120))
freude_single_per_sent = dict((l,list()) for l in range(120))

angst_multi_full_per_sent = dict((k,list()) for k in range(120))
freude_multi_full_per_sent = dict((l,list()) for l in range(120))

angst_multi_tsne_per_sent = dict((k,list()) for k in range(120))
freude_multi_tsne_per_sent = dict((l,list()) for l in range(120))





for i in range(len(harry_data)):
    for key in set(angst_single[1].index.values):
        if key in harry_data.loc[i]['nouns']:
            try:
                angst_single_per_sent[i].append(angst_single[1][key][0])
                freude_single_per_sent[i].append(freude_single[1][key][0])
                
                angst_multi_full_per_sent[i].append(angst_multi_full[1][key][0])
                freude_multi_full_per_sent[i].append(freude_multi_full[1][key][0])
                
                angst_multi_tsne_per_sent[i].append(angst_multi_tsne[1][key][0])
                freude_multi_tsne_per_sent[i].append(freude_multi_tsne[1][key][0])
            
            except:
                angst_single_per_sent[i].append(angst_single[1][key])
                freude_single_per_sent[i].append(freude_single[1][key])
                
                angst_multi_full_per_sent[i].append(angst_multi_full[1][key])
                freude_multi_full_per_sent[i].append(freude_multi_full[1][key])
                
                angst_multi_tsne_per_sent[i].append(angst_multi_tsne[1][key])
                freude_multi_tsne_per_sent[i].append(freude_multi_tsne[1][key])


for key in angst_single_per_sent.keys():
    angst_single_per_sent[key] = np.mean(angst_single_per_sent[key])
    freude_single_per_sent[key] = np.mean(freude_single_per_sent[key])
    
    angst_multi_full_per_sent[key] = np.mean(angst_multi_full_per_sent[key])
    freude_multi_full_per_sent[key] = np.mean(freude_multi_full_per_sent[key])
    
    angst_multi_tsne_per_sent[key] = np.mean(angst_multi_tsne_per_sent[key])
    freude_multi_tsne_per_sent[key] = np.mean(freude_multi_tsne_per_sent[key])


angst_1 = pd.Series(index = angst_single_per_sent.keys(), data=angst_single_per_sent.values())
freude_1 = pd.Series(index = freude_single_per_sent.keys(), data=freude_single_per_sent.values())

angst_full = pd.Series(index = angst_multi_full_per_sent.keys(), data=angst_multi_full_per_sent.values())
freude_full = pd.Series(index = freude_multi_full_per_sent.keys(), data=freude_multi_full_per_sent.values())

angst_tsne = pd.Series(index = angst_multi_tsne_per_sent.keys(), data=angst_multi_tsne_per_sent.values())
freude_tsne = pd.Series(index = freude_multi_tsne_per_sent.keys(), data=freude_multi_tsne_per_sent.values())



### Investigation of Correlations between HUMAN RATINGS and PREDICTIONS

# SINGLE_LABEL

print(np.corrcoef(harry_data['Mean_Fear_RATING'].values, angst_1.values))

print(np.corrcoef(harry_data['Mean_happiness_RATING'].values, freude_1.values))


plt.scatter(harry_data['Mean_Fear_RATING'].values, angst_1.values)
plt.scatter(harry_data['Mean_happiness_RATING'].values, freude_1.values)


# MULTILABEL 300-dim

print(np.corrcoef(harry_data['Mean_Fear_RATING'].values, angst_full.values))

print(np.corrcoef(harry_data['Mean_happiness_RATING'].values, freude_full.values))


plt.scatter(harry_data['Mean_Fear_RATING'].values, angst_full.values)
plt.scatter(harry_data['Mean_happiness_RATING'].values, freude_full.values)


# MULTILABEL TSNE

print(np.corrcoef(harry_data['Mean_Fear_RATING'].values, angst_tsne.values))

print(np.corrcoef(harry_data['Mean_happiness_RATING'].values, freude_tsne.values))


plt.scatter(harry_data['Mean_Fear_RATING'].values, angst_tsne.values)
plt.scatter(harry_data['Mean_happiness_RATING'].values, freude_tsne.values)




### CLASSIFICATION TASK

category_predictions = pd.DataFrame(index=range(120), columns=['single_label_pred_CATEGORY', 
                                                               'multilabel_full_pred_CATEGORY','multilabel_tsne_pred_CATEGORY', 'True Category'])

##### HP human rating categorization 
import seaborn as sns

sns.set()
fig,ax = plt.subplots(figsize=(12,7))
line1 = plt.plot(range(120),harry_data['Mean_Fear_RATING'].values,color='blue')
line2 = plt.plot(range(120),harry_data['Mean_happiness_RATING'].values,color='green')
ax.legend(['Angst','Freude'])

plt.axvline(x=np.where(harry_data['Category'].values == 'H')[0][0], color='k', linestyle='--', lw=1)
plt.axvline(x=np.where(harry_data['Category'].values == 'N')[0][0], color='k', linestyle='--',lw=1)

ax.set_xlabel('Sentence Number',size=15)
ax.set_ylabel('human Rating',size=15)


##### actual classification

def category_extractor(angst_series, freude_series, stds=1, method='std'):
    '''
    extracts the PREDICTED Category according to the following rule:
        1. exttract the higher rating (absolute)
        2. investigate if the higher rating is larger than the mean + (std)* standard_deviation of the respective array
        3. if yes, then it defines the Category (irrespective of the other value
        4. if no, the category is neutral
    '''
    
    category_list = []
    
    if method == 'std':

        for (a,f) in zip(angst_series, freude_series):
            if a>f:
                if a > (np.mean(angst_series)+stds*np.std(angst_series)):
                    category_list.append('F')
                else:
                    category_list.append('N')
                
            else:
                if f > (np.mean(freude_series)+stds*np.std(freude_series)):
                    category_list.append('H')
                else:
                    category_list.append('N')
    
    return category_list
        


single = category_extractor(angst_1, freude_1)
multi_full = category_extractor(angst_full, freude_full)
multi_tsne = category_extractor(angst_tsne, freude_tsne)

category_predictions['single_label_pred_CATEGORY'] = single
category_predictions['multilabel_full_pred_CATEGORY'] = multi_full
category_predictions['multilabel_tsne_pred_CATEGORY'] = multi_tsne
category_predictions['True Category'] = harry_data['Category']


from sklearn.metrics import confusion_matrix


classification_singlelabel = confusion_matrix(category_predictions['single_label_pred_CATEGORY'].values, 
                                             category_predictions['True Category'].values)


classification_multilabel_full = confusion_matrix(category_predictions['multilabel_full_pred_CATEGORY'].values,
                                             category_predictions['True Category'].values)

classification_multilabel_tsne = confusion_matrix(category_predictions['multilabel_tsne_pred_CATEGORY'].values,
                                             category_predictions['True Category'].values)
                                        




import itertools
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    sns.set_style('white')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],2)
        
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 4.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



##### Matrix for singlelabel classification (+ F1 and accuracy)
plot_confusion_matrix(classification_singlelabel, classes=list(set(category_predictions['True Category'].values)), 
                      normalize=True, title='Singlelabel')

print(accuracy_score(category_predictions['single_label_pred_CATEGORY'].values, 
                                             category_predictions['True Category'].values))
print(f1_score(category_predictions['single_label_pred_CATEGORY'].values, 
                                             category_predictions['True Category'].values, average='macro'))



##### Matrix for multilabel classification with full model (+ F1 and accuracy)
plot_confusion_matrix(classification_multilabel_full, classes=list(set(category_predictions['True Category'].values)), 
                      normalize=True, title='Multilabel full VSM')

print(accuracy_score(category_predictions['multilabel_full_pred_CATEGORY'].values,
                                             category_predictions['True Category'].values))
print(f1_score(category_predictions['multilabel_full_pred_CATEGORY'].values, 
                                             category_predictions['True Category'].values, average='macro'))



##### Matrix for multilabel classification with tSNE model (+ F1 and accuracy)
plot_confusion_matrix(classification_multilabel_tsne, classes=list(set(category_predictions['True Category'].values)), 
                      normalize=True, title='Multilabel tSNE')

print(accuracy_score(category_predictions['multilabel_tsne_pred_CATEGORY'].values,
                                             category_predictions['True Category'].values))
print(f1_score(category_predictions['multilabel_tsne_pred_CATEGORY'].values, 
                                             category_predictions['True Category'].values, average='macro'))

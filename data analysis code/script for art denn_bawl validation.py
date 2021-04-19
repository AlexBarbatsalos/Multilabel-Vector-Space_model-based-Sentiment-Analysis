import pandas as pd
import os
import numpy as np
import nltk
import math
import sklearn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

wd = ''         ### INSERT WORKING DIRECTORY HERE

os.chdir(wd)

import SentiArt



from gensim.models import KeyedVectors
model = 'C:\\Users\\Anwender\\Documents\\SCAN\\Masterarbeitsthema\\Jacobs Machine Learning Linguistics\
\\Vector Space Models\\gensim models\\120sdewac_sg300.vec'

vsm = KeyedVectors.load_word2vec_format(model)



#### Multilabel Setup

no_sublabels = 200

label_frame = pd.DataFrame(index=range(no_sublabels), columns = ['Angst', 'Trauer', 'Freude','Wut','Ãœberraschung','Ekel'])
just_labels = pd.DataFrame(index=range(no_sublabels), columns = ['Angst', 'Trauer', 'Freude','Wut','Ãœberraschung','Ekel'])

for col in label_frame.columns:
    label_frame.iloc[:][col] = vsm.most_similar(positive=[col], topn=no_sublabels)
    just_labels.iloc[:][col] = [element[0] for element in vsm.most_similar(positive=[col], topn=no_sublabels)]





### setup of sentiArt-table and DENN_BAWL import


fn = 'DENN-BAWL_AJ14.txt'
db = pd.read_csv(fn, decimal=",", encoding='utf-8')

db['up'] = db['up'].replace([el for el in db['up'].values], [el.lower() for el in db['up'].values])


words = list(db['up'])


man = SentiArt.SentiArt_Manager(words, None)
man.load_model('Vector Space Models\\gensim models\\120sdewac_sg300.vec')





### DENN_BAWL item analysis and exclusion

ocw = {'wut':[], 'freude':[], 'trauer':[], 'angst':[],'ekel':[]}


for col in ['wut sd', 'freude sd', 'trauer sd', 'angst sd', 'ekel sd']:
    q75 = np.percentile(db[col],75)
    ocw[col.split()[0]] = db['up'][db[col]<q75].values
    
    

shared = []
for word in db['up'].values:
    if word in ocw['wut'] and word in ocw['trauer'] and word in ocw['freude'] and word in ocw['angst'] and word in ocw['ekel']:
        shared.append(word)





### Multiple Linear Regression (full VSM-model)

### multilabel_columns dicts
angst = {'Angst':list(just_labels['Angst'])}
ekel = {'Ekel':list(just_labels['Ekel'])}
wut = {'Wut':list(just_labels['Wut'])}
freude = {'Freude':list(just_labels['Freude'])}
trauer = {'Trauer':list(just_labels['Trauer'])}



###single label column dicts
angst_single = {'Angst':['Angst']}
ekel_single = {'Ekel':['Ekel']}
wut_single = {'Wut':['Wut']}
freude_single = {'Freude':['Freude']}
trauer_single = {'Trauer':['Trauer']}




#### setting up 2 new columns for classification 



for row in db.index:
    db.loc[row,('MAXLABEL_VALUE_target')] = max(db.loc[row]['freude'],db.loc[row]['angst'],db.loc[row]['wut'],
                                  db.loc[row]['ekel'], db.loc[row]['trauer'])
    

    
value_frame = pd.DataFrame(index = db.index, columns=[db.columns.values[18],db.columns.values[20],db.columns.values[22],db.columns.values[24],
                                                     db.columns.values[26]], data = np.transpose(np.asarray([db['freude'].values, 
                                                                                                             db['wut'].values, db['trauer'].values,
                                                                                                                          db['angst'].values, db['ekel'].values])))
db['MAXLABEL_target'] = value_frame.idxmax(axis=1, skipna=True)






#### multilabel columns

angst_col = man.compute_single_column(ocw['angst'],angst)
ekel_col = man.compute_single_column(ocw['ekel'],ekel)
trauer_col = man.compute_single_column(ocw['trauer'],trauer)
freude_col = man.compute_single_column(ocw['freude'],freude)
wut_col = man.compute_single_column(ocw['wut'],wut)



### single label columns
angst_col_single = man.compute_single_column(ocw['angst'],angst_single)
ekel_col_single = man.compute_single_column(ocw['ekel'],ekel_single)
trauer_col_single = man.compute_single_column(ocw['trauer'],trauer_single)
freude_col_single = man.compute_single_column(ocw['freude'],freude_single)
wut_col_single = man.compute_single_column(ocw['wut'],wut_single)




def validation_preparation(validation_table, sentiart_col_multi, sentiart_col_single, col_name='Column', merge_col='up'):
    '''
    inputs: validation_table, sentiart_col(prealigned), label_name of the column of validation table,
            which is used for alignment (default='word'), name of new col in frame
    output: merged df of validation_table and sentiart_col
    
    this function adds two new columns to the db, one multilabel (col_name) and one with the corresponding 
    single_label (see variable 'single_string1')
    '''
    
    single_string = col_name + ' single'
    frame = pd.DataFrame(index = range(len(sentiart_col_multi)), columns = [col_name, single_string, merge_col])
    frame[col_name] = sentiart_col_multi.values
    
    frame[single_string] = sentiart_col_single.values

        
    frame[merge_col] = sentiart_col_multi.index
    
    try:
        merged_frame = validation_table.merge(frame, on=merge_col, how='left')
    
    except:
        print('No column named "word" in validation_frame')
        return None
        
    merged_frame_act = merged_frame.sort_values(col_name)[:len(sentiart_col_multi)]
    
    return merged_frame_act





frame_angst = validation_preparation(db, angst_col[1], sentiart_col_single = angst_col_single[1], col_name = 'angst__values')
frame_ekel = validation_preparation(db, ekel_col[1], sentiart_col_single=ekel_col_single[1], col_name='ekel__values')
frame_trauer = validation_preparation(db, trauer_col[1], sentiart_col_single=trauer_col_single[1], col_name='trauer__values')
frame_freude = validation_preparation(db, freude_col[1], sentiart_col_single=freude_col_single[1], col_name='freude__values')
frame_wut = validation_preparation(db, wut_col[1], sentiart_col_single=wut_col_single[1], col_name='wut__values')






def compute_z_standardized_cosine_similarites(label_list, words_of_interest):
    '''
    computes z_standardized cosine similarities for single labels for a given set of words
    based on the vsm within the SentiArt_Manager class --> so import model first
    
    '''
    global default_matrix
    
    default_matrix = np.zeros((len(words_of_interest),len(label_list)))
    
    for i in range(len(words_of_interest)):
        for j in range(len(label_list)):
            try:
                default_matrix[i,j] = man.model.similarity(words_of_interest[i],label_list[j])
            except:
                default_matrix[i,j] = man.model.similarity(words_of_interest[i].capitalize(),label_list[j])
    
    ## compute means and standard deviations for all labels given the filled matrix
    means_and_stds_per_label = {'means': np.zeros(len(label_list)), 'stds': np.zeros(len(label_list))}
    
    for l in range(len(label_list)):
        means_and_stds_per_label['means'][l]= np.mean(default_matrix[:,l])
        means_and_stds_per_label['stds'][l]= np.std(default_matrix[:,l])
    
    standardized_matrix = np.zeros((default_matrix.shape))
    
    for i in range(len(words_of_interest)):
        for j in range(len(label_list)):
            standardized_matrix[i,j] = (default_matrix[i,j] - means_and_stds_per_label['means'][j])/means_and_stds_per_label['stds'][j]
            
    
    return standardized_matrix



#### X (predictor matrix) and y (value-vector to be predicted) for labels EKEL, WUT, FREUDE, TRAURIGKEIT, ANGST

ekel_sublabels = list(just_labels['Ekel'])
ekel_sublabel_matrix = compute_z_standardized_cosine_similarites(ekel_sublabels, ocw['ekel'])
ekel_ratings = np.asarray(frame_ekel['ekel'])


wut_sublabels = list(just_labels['Wut'])
wut_sublabel_matrix = compute_z_standardized_cosine_similarites(wut_sublabels, ocw['wut'])
wut_ratings = np.asarray(frame_wut['wut'])


freude_sublabels = list(just_labels['Freude'])
freude_sublabel_matrix = compute_z_standardized_cosine_similarites(freude_sublabels, ocw['freude'])
freude_ratings = np.asarray(frame_freude['freude'])


trauer_sublabels = list(just_labels['Trauer'])
trauer_sublabel_matrix = compute_z_standardized_cosine_similarites(trauer_sublabels, ocw['trauer'])
trauer_ratings = np.asarray(frame_trauer['trauer'])


angst_sublabels = list(just_labels['Angst'])
angst_sublabel_matrix = compute_z_standardized_cosine_similarites(angst_sublabels, ocw['angst'])
angst_ratings = np.asarray(frame_angst['angst'])






def label_extractor(predictor_matrix, true_vector, label_dict, label_name, no_sublabels, model=man):
    '''
    label_extractor runs a multiple LinReg given:
    (1) a given model
    (2) a certain set of sublabels given a discrete emotion label
    (3) A predictor matrix with VSM-based predictions
    (4) A true vector which is to be predicted
    
    It extracts (no_sublabels) sublabels from the initial sublabel set based on the beta-weights in the LinReg.
    '''
    from sklearn.linear_model import LinearRegression

    #set up LinReg model
    reg_model = LinearRegression()
    reg_model.fit(predictor_matrix, true_vector)

    # extract high-weight sublabels
    weights_per_label = pd.Series(index = list(label_dict.values), data=reg_model.coef_)
    weights = weights_per_label.sort_values(ascending=False)
    sublabels_reg ={label_name: list(label_name) + list(weights.index[:no_sublabels])
    

    senti_col = model.compute_single_column(ocw[label_name.lower())], sublabels_reg)
    senti_col_single = model.compute_single_column(ocw[label_name.lower())], {label_name: list(label_name)}
    frame = validation_preparation(db, senti_col[1], senti_col_single[1], col_name=label_name +'_reg', merge_col='up')

    return frame


#### lin Reg for ANGST full

from sklearn.linear_model import LinearRegression

reg_model = LinearRegression()

reg_model.fit(angst_sublabel_matrix, angst_ratings)

angst_score = reg_model.score(angst_sublabel_matrix, angst_ratings)

weights_per_label = pd.Series(index = angst_sublabels, data=reg_model.coef_)
weights = weights_per_label.sort_values(ascending=False)
angst_sublabels_reg = list(['Angst']+list(weights.index[:25]))  ### 26 basierend auf kurve!!!
angst_reg_full = {'Angst':angst_sublabels_reg}



angst__reg_full = man.compute_single_column(ocw['angst'],angst_reg_full)
frame_angst = validation_preparation(db, angst__reg_full[1], angst_col_single[1], col_name='angst_reg', merge_col='up')

print(np.corrcoef(frame_angst['angst'], frame_angst['angst_reg single']))
print(np.corrcoef(frame_angst['angst'].values,frame_angst['angst_reg'].values))
print(np.corrcoef(frame_angst['V'],frame_angst['angst_reg']))

plt.scatter(frame_angst['angst'].values,frame_angst['angst_reg'])






#### ANGST lin Reg tsne

from sklearn.linear_model import LinearRegression



angst_ratings = angst['angst_x']
angst_predictors = np.asarray(angst.loc[:,angst.columns.values[np.where(angst.columns.values == 'Angst')[0][0]]:angst.columns.values[-1]].values)

reg_model_angst_tsne = LinearRegression()

reg_model_angst_tsne.fit(angst_predictors, angst_ratings)




weights_per_label = pd.Series(index = angst.columns.values[np.where(angst.columns.values == 'Angst')[0][0]:], data=reg_model_angst_tsne.coef_)
weights = weights_per_label.sort_values(ascending=False)
angst_sublabels_reg = list(['Angst']+list(weights.index[:20]))

angst_reg = {'Angst':angst_sublabels_reg}


angst__reg_tsne = man.compute_single_column(ocw['angst'],angst_reg)
frame_angst = validation_preparation(db, angst__reg_tsne[1], angst_col_single[1], col_name='angst_reg')

print(np.corrcoef(frame_angst['angst'], frame_angst['angst_reg single']))
print(np.corrcoef(frame_angst['angst'].values,frame_angst['angst_reg'].values))
print(np.corrcoef(frame_angst['V'],frame_angst['angst_reg']))

plt.scatter(frame_angst['angst'].values,frame_angst['angst_reg'])


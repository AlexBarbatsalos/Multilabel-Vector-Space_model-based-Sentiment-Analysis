import os
import numpy as np
import nltk
import math
import sklearn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

working_dir = 'C:\\Users\\Anwender\\Documents\\SCAN\Masterarbeitsthema\\Jacobs Machine Learning Linguistics'
os.chdir(working_dir)

import SentiArt


fn = 'DENN-BAWL_AJ14.txt'  ## Validation dataset
db = pd.read_csv(fn, decimal=",", encoding='utf-8')

# make all words in DENN BAWL lowercase
db['up'] = db['up'].replace([el for el in db['up'].values], [el.lower() for el in db['up'].values])


words = list(db['up'])


man = SentiArt.SentiArt_Manager(words, None)
man.load_model(model_dir)



### creating initial label set

no_sublabels = 200


label_frame = pd.DataFrame(index=range(no_sublabels), columns = ['Angst', 'Trauer', 'Freude','Wut','Ekel'])
just_labels = pd.DataFrame(index=range(no_sublabels), columns = ['Angst', 'Trauer', 'Freude','Wut','Ekel'])

for col in label_frame.columns:
    label_frame.iloc[:][col] = man.model.most_similar(positive=[col], topn=no_sublabels)
    just_labels.iloc[:][col] = [element[0] for element in man.model.most_similar(positive=[col], topn=no_sublabels)]




### DENN_BAWL item analysis

# outlier cleaned words
ocw = {'wut':[], 'freude':[], 'trauer':[], 'angst':[],'ekel':[]}
quartiles_75 = []

for col in ['wut sd', 'freude sd', 'trauer sd', 'angst sd', 'ekel sd']:
    q75 = np.percentile(db[col],75)
    quartiles_75.append(q75)
    ocw[col.split()[0]] = db['up'][db[col]<q75].values
    
    

shared = []
for word in db['up'].values:
    if word in ocw['wut'] and word in ocw['trauer'] and word in ocw['freude'] and word in ocw['angst'] and word in ocw['ekel']:
        shared.append(word)



   
### item analysis figure ['item analysis DENN BAWL Ratings.png']

db_cols =['wut sd', 'freude sd', 'trauer sd', 'angst sd', 'ekel sd']
color_code = ['red','green','orange','blue','purple']
sns.set()

fig = plt.figure(figsize=(20,10))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1,6):
    plt.subplot(2,3,i)
    plt.hist(db[db_cols[i-1]], bins=15, color=color_code[i-1], alpha=0.5)
    plt.text(0.2,260,s=str(len(ocw[db_cols[i-1].split()[0]]))+' words',style='italic',bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
    plt.axis('tight')
    plt.axvline(quartiles_75[i-1],0,300,label='q_75', color='red', linestyle='dashed')
    plt.title(str.capitalize(db_cols[i-1].split()[0]), loc='left', fontsize=20)
    plt.xlabel('Rating Standard Deviation')
    plt.ylabel('Number of Words')
    plt.ylim(0,300)
    plt.xlim(0,1.75)
    plt.legend(loc='upper right')
    
    

### Preparations for Multiple Linear Regression Extractor and Classification

# multilabel_columns dicts
angst = {'Angst':list(just_labels['Angst'])}
ekel = {'Ekel':list(just_labels['Ekel'])}
wut = {'Wut':list(just_labels['Wut'])}
freude = {'Freude':list(just_labels['Freude'])}
trauer = {'Trauer':list(just_labels['Trauer'])}

#single label column dicts
angst_single = {'Angst':['Angst']}
ekel_single = {'Ekel':['Ekel']}
wut_single = {'Wut':['Wut']}
freude_single = {'Freude':['Freude']}
trauer_single = {'Trauer':['Trauer']}

# setting up 2 new columns for classification 

for row in db.index:
    db.loc[row,('MAXLABEL_VALUE_target')] = max(db.loc[row]['freude'],db.loc[row]['angst'],db.loc[row]['wut'],
                                  db.loc[row]['ekel'], db.loc[row]['trauer'])
    

    
value_frame = pd.DataFrame(index = db.index, columns=[db.columns.values[18],db.columns.values[20],db.columns.values[22],db.columns.values[24],
                                                     db.columns.values[26]], data = np.transpose(np.asarray([db['freude'].values, 
                                                                                                             db['wut'].values, db['trauer'].values,
                                                                                                                          db['angst'].values, db['ekel'].values])))
db['MAXLABEL_target'] = value_frame.idxmax(axis=1, skipna=True)







### LinReg Label Extraction


# multilabel columns

angst_col = man.compute_single_column(ocw['angst'],angst)
ekel_col = man.compute_single_column(ocw['ekel'],ekel)
trauer_col = man.compute_single_column(ocw['trauer'],trauer)
freude_col = man.compute_single_column(ocw['freude'],freude)
wut_col = man.compute_single_column(ocw['wut'],wut)



# single label columns
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



frame_angst = validation_preparation(db, angst_col[1], sentiart_col_single = angst_col_single[1], col_name = 'angst__values')
frame_ekel = validation_preparation(db, ekel_col[1], sentiart_col_single=ekel_col_single[1], col_name='ekel__values')
frame_trauer = validation_preparation(db, trauer_col[1], sentiart_col_single=trauer_col_single[1], col_name='trauer__values')
frame_freude = validation_preparation(db, freude_col[1], sentiart_col_single=freude_col_single[1], col_name='freude__values')
frame_wut = validation_preparation(db, wut_col[1], sentiart_col_single=wut_col_single[1], col_name='wut__values')



# X (predictor matrix) and y (value-vector to be predicted) for labels EKEL, WUT, FREUDE, TRAUER, ANGST

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



def label_extractor(predictor_matrix, true_vector, label_list, label_name, no_sublabels=15, model=man):
    ''' This function runs a multiple linear Regression on a set of words (encoded in the predictor matrix) with a number of sublabels
        for a given emotion (label_name). It extracts (no_sublabels) sublabels with the highest beta-weights
        from the Regression and summarizes them in a dict.'''
    from sklearn.linear_model import LinearRegression

    #set up LinReg model
    reg_model = LinearRegression()
    reg_model.fit(predictor_matrix, true_vector)

    # extract high-weight sublabels
    global weights_per_label
    weights_per_label = pd.Series(index = label_list, data=reg_model.coef_)
    weights = weights_per_label.sort_values(ascending=False)
    sublabels_reg ={label_name: [label_name] + list(weights.index[:no_sublabels-1])}
    

    return sublabels_reg

## lin Reg for ANGST(full)

angst_reg_full = label_extractor(angst_sublabel_matrix, angst_ratings, angst_sublabels,'Angst')

### eye validity cleaning has to be applied
#print(angst_reg_full)
#angst_reg_full['Angst'].remove(....)


angst__reg_full = man.compute_single_column(ocw['angst'],angst_reg_full)
frame_angst = validation_preparation(db, angst__reg_full[1], angst_col_single[1], col_name='angst_reg', merge_col='up')

print(np.corrcoef(frame_angst['angst'], frame_angst['angst_reg single']))
print(np.corrcoef(frame_angst['angst'].values,frame_angst['angst_reg'].values))




## lin Reg for EKEL(full)

ekel_reg_full = label_extractor(ekel_sublabel_matrix, ekel_ratings, ekel_sublabels,'Ekel')

### eye validity cleaning has to be applied
#print(ekel_reg_full)
#ekel_reg_full['Ekel'].remove(....)

ekel__reg_full = man.compute_single_column(ocw['ekel'],ekel_reg_full)
frame_ekel = validation_preparation(db, ekel__reg_full[1], ekel_col_single[1], col_name='ekel_reg')

print(np.corrcoef(frame_ekel['ekel'], frame_ekel['ekel_reg single']))
print(np.corrcoef(frame_ekel['ekel'],frame_ekel['ekel_reg']))





## lin Reg for FREUDE(full)

freude_reg_full = label_extractor(freude_sublabel_matrix, freude_ratings, freude_sublabels,'Freude')

### eye validity cleaning has to be applied
#print(freude_reg_full)
#freude_reg_full['Freude'].remove(....)

freude__reg_full = man.compute_single_column(ocw['freude'],freude_reg_full)
frame_freude = validation_preparation(db, freude__reg_full[1], freude_col_single[1], col_name='freude_reg')

print(np.corrcoef(frame_freude['freude'], frame_freude['freude_reg single']))
print(np.corrcoef(frame_freude['freude'],frame_freude['freude_reg']))



## lin Reg for TRAUER(full)

trauer_reg_full = label_extractor(trauer_sublabel_matrix, trauer_ratings, trauer_sublabels, 'Trauer')

### eye validity cleaning has to be applied
#print(trauer_reg_full)
#trauer_reg_full['Trauer'].remove(....)


trauer__reg_full = man.compute_single_column(ocw['trauer'],trauer_reg_full)
frame_trauer = validation_preparation(db, trauer__reg_full[1], trauer_col_single[1], col_name='trauer_reg')

print(np.corrcoef(frame_trauer['trauer'], frame_trauer['trauer_reg single']))
print(np.corrcoef(frame_trauer['trauer'],frame_trauer['trauer_reg']))


## lin Reg for WUT(full)

wut_reg_full = label_extractor(wut_sublabel_matrix, wut_ratings, wut_sublabels, 'Wut')

### eye validity cleaning has to be applied
#print(wut_reg_full)
#wut_reg_full['Wut'].remove(....)

wut__reg_full = man.compute_single_column(ocw['wut'],wut_reg_full)
frame_wut = validation_preparation(db, wut__reg_full[1], wut_col_single[1], col_name='wut_reg')

print(np.corrcoef(frame_wut['wut'], frame_wut['wut_reg single']))
print(np.corrcoef(frame_wut['wut'],frame_wut['wut_reg']))

### weights per label figure ['beta weight distribution across discrete emotion labels.png']


sns.set()
color_code = ['red','green','orange','blue','purple']
emotion_dict = {'red':weights_per_label_wut, 'green':weights_per_label_freude, 'orange':weights_per_label_trauer,
                'blue': weights_per_label_angst, 'purple':weights_per_label_ekel}
table_names = ['Wut', 'Freude','Trauer', 'Angst','Ekel']

fig = plt.figure(figsize=(20,10))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1,6):
    plt.subplot(2,3,i)
    plt.plot(emotion_dict[list(emotion_dict.keys())[i-1]].sort_values(ascending=False).values, color=list(emotion_dict.keys())[i-1], linewidth=2)
    x0 = 14
    y0=emotion_dict[list(emotion_dict.keys())[i-1]].sort_values(ascending=False).values[14]
    plt.plot(x0, y0, "d",color='black',markersize=10, alpha=0.6)
    plt.axvline(14, linestyle='dashed',color='black',alpha=0.6)
    plt.xlabel('sublabel')
    plt.ylabel('beta weight')
    plt.title(table_names[i-1], loc='left', fontsize=20)
    
    
    
### TSNE based label extraction

# input for tSNE or PCA reduction must be a matrix with no_vsm_dimensions x no_words dimensions
#preparation for multiindexed df
labels = ['Angst']*no_sublabels + ['Trauer']*no_sublabels + ['Freude']*no_sublabels + ['Wut']*no_sublabels + ['Ekel']*no_sublabels + ['words']*len(man.table_words)
sublabels = ['Angst']+list(just_labels.iloc[:-1]['Angst']) + ['Trauer'] + list(just_labels.iloc[:-1]['Trauer']) + ['Freude']+list(just_labels.iloc[:-1]['Freude']) + ['Wut']+list(just_labels.iloc[:-1]['Wut']) + ['Ekel']+ list(just_labels.iloc[:-1]['Ekel']) + list(man.table_words) 

label_tuples = list(zip(labels,sublabels))
ind = pd.MultiIndex.from_tuples(label_tuples, names=['labels', 'sublabels'])

#set up and populate df
vsm_frame = pd.DataFrame(index = ind, columns = range(1,301))
for row in vsm_frame.index:
    try:
        vsm_frame.loc[row] = man.model[row[1]]
    except:
        vsm_frame.loc[row] = man.model[row[1].capitalize()]


import time

start = time.time()
tsne = TSNE(n_components=2, n_iter=4000)
tsne.fit_transform(pca_reduced.values)
end = time.time()

print((end-start)/60)

#the .embedding_ method returns a vectorized form of the n_component-dimensional reduction
tsne.embedding_.shape

embedded_frame = pd.DataFrame(data = tsne.embedding_, index = ind, columns = ['dim 1', 'dim 2'])

#for excel table see /data
embedded_frame.to_excel('2D tsne embedded labels and denn_bawl_words.xlsx')  #save frame for fast retrieval
embedded = pd.read_excel(working_dir, index_col=[0,1], engine='openpyxl')



def similarity(a,b, method='cosine'):
    '''
    computes the similarity between two n-dimensional objects
    
    inputs are two n-dimensional arrays
    
    method: 'cosine'(default) or 'euclidean'
    
   
    '''
    
    import math
    
    result = None
    
    if method == 'cosine':
        
        result=(a.dot(b))/(math.sqrt(sum(a**2))*math.sqrt(sum(b**2)))
    
    elif method == 'euclidean':
        
        result = math.sqrt(sum((a-b)**2))
        
        
    else:
        print('No method named', method, 'try "cosine" or "euclidean"')
    
        
    
    return result
    


def z_standardize(similarity_frame):
    '''
    standardizes a similarity frame, i.e. an indexed similarity matrix of n(rows)*m(columns) entries
    
    in the current context of vsms,
    m is the number of sublabels and
    n is the number of words_of_interest (e.g. denn_bawl_words)
    '''
    
    mean_per_label = dict()
    std_per_label = dict()
    
    ## new frame to accumulate for better transparency
    stand_frame = pd.DataFrame(index=similarity_frame.index, columns = similarity_frame.columns)
    
    for col in similarity_frame.columns.values:
        # this computes means and stds per sublabel over words of interest
        mean_per_label[col] = np.mean(similarity_frame[col])
        std_per_label[col] = np.std(similarity_frame[col])
        
        # the inner loop itertates over all entries in col and computes z_standardized values
        for index in similarity_frame.index.values:
            stand_frame.loc[index][col] = (similarity_frame.loc[index][col]-mean_per_label[col])/std_per_label[col]
    
    
    
    return stand_frame





## TSNE-based multilabel sets for ANGST, WUT, TRAUER, FREUDE, EKEL

# compute frame for ANGST
sim_frame_angst = pd.DataFrame(index=ocw['angst'], columns=embedded.loc['Angst'].index)

for ind in sim_frame_angst.index.values:
    for col in sim_frame_angst.columns.values:
        sim_frame_angst.loc[ind][col] = similarity(embedded.loc[('words',ind)],
                                                   embedded.loc[('Angst',col)], method='euclidean')

standardized_angst = z_standardize(sim_frame_angst)

# see /data
standardized_angst.to_excel(working_dir)


# compute frame for EKEL
sim_frame_ekel = pd.DataFrame(index=ocw['ekel'], columns=embedded.loc['Ekel'].index)

for ind in sim_frame_ekel.index.values:
    for col in sim_frame_ekel.columns.values:
        sim_frame_ekel.loc[ind][col] = similarity(embedded.loc[('words',ind)],
                                                   embedded.loc[('Ekel',col)], method='euclidean')

standardized_ekel = z_standardize(sim_frame_ekel)

#see /data
standardized_ekel.to_excel(working_dir)


# compute frame for FREUDE
sim_frame_freude = pd.DataFrame(index=ocw['freude'], columns=embedded.loc['Freude'].index)

for ind in sim_frame_freude.index.values:
    for col in sim_frame_freude.columns.values:
        sim_frame_freude.loc[ind][col] = similarity(embedded.loc[('words',ind)],
                                                   embedded.loc[('Freude',col)], method='euclidean')

standardized_freude = z_standardize(sim_frame_freude)

#see /data
standardized_freude.to_excel(working_dir)


# compute frame for WUT
sim_frame_wut = pd.DataFrame(index=ocw['wut'], columns=embedded.loc['Wut'].index)

for ind in sim_frame_wut.index.values:
    for col in sim_frame_wut.columns.values:
        sim_frame_wut.loc[ind][col] = similarity(embedded.loc[('words',ind)],
                                                   embedded.loc[('Wut',col)], method='euclidean')

standardized_wut = z_standardize(sim_frame_wut)

#see /data
standardized_wut.to_excel(working_dir)


# compute frame for TRAUER
sim_frame_trauer = pd.DataFrame(index=ocw['trauer'], columns=embedded.loc['Trauer'].index)

for ind in sim_frame_trauer.index.values:
    for col in sim_frame_trauer.columns.values:
        sim_frame_trauer.loc[ind][col] = similarity(embedded.loc[('words',ind)],
                                                   embedded.loc[('Trauer',col)], method='euclidean')

standardized_trauer = z_standardize(sim_frame_trauer)

# see /data
standardized_trauer.to_excel(working_dir)


# load frames and merge with DENN_BAWL for follow_up LinReg
stand_tsne_angst = pd.read_excel('tSNE angst frame.xlsx', engine='openpyxl')
stand_tsne_ekel = pd.read_excel('tSNE ekel frame.xlsx', engine='openpyxl')
stand_tsne_freude = pd.read_excel('tSNE freude frame.xlsx', engine='openpyxl')
stand_tsne_wut = pd.read_excel('tSNE wut frame.xlsx', engine='openpyxl')
stand_tsne_trauer = pd.read_excel('tSNE trauer frame.xlsx', engine='openpyxl')

stand_tsne_angst.columns.values[0] = 'up'
stand_tsne_ekel.columns.values[0] = 'up'
stand_tsne_freude.columns.values[0] = 'up'
stand_tsne_wut.columns.values[0] = 'up'
stand_tsne_trauer.columns.values[0] = 'up'



angst_all = db.merge(stand_tsne_angst,on='up',how='left').sort_values('Angst')
angst = angst_all[:np.where(np.isnan(angst_all['Angst']))[0][0]]

ekel_all = db.merge(stand_tsne_ekel,on='up',how='left').sort_values('Ekel')
ekel = ekel_all[:np.where(np.isnan(ekel_all['Ekel']))[0][0]]

freude_all = db.merge(stand_tsne_freude,on='up',how='left').sort_values('Freude')
freude = freude_all[:np.where(np.isnan(freude_all['Freude']))[0][0]]

wut_all = db.merge(stand_tsne_wut,on='up',how='left').sort_values('Wut')
wut = wut_all[:np.where(np.isnan(wut_all['Wut']))[0][0]]

trauer_all = db.merge(stand_tsne_trauer,on='up',how='left').sort_values('Trauer')
trauer = trauer_all[:np.where(np.isnan(trauer_all['Trauer']))[0][0]]




## LinReg for ANGST (TSNE)

angst_ratings = angst['angst_x']
angst_predictors = np.asarray(angst.loc[:,angst.columns.values[np.where(angst.columns.values == 'Angst')[0][0]]:angst.columns.values[-1]].values)

angst_reg1 = label_extractor(angst_predictors, angst_ratings, stand_tsne_angst.columns.values[1:],'Angst',no_sublabels=19)
weights_per_label_angst_tsne = weights_per_label
#Eye validity_check (see above)

angst__reg_tsne = man.compute_single_column(ocw['angst'],angst_reg1)
frame_angst = validation_preparation(db, angst__reg_tsne[1], angst_col_single[1], col_name='angst_reg')

print(np.corrcoef(frame_angst['angst'], frame_angst['angst_reg single']))
print(np.corrcoef(frame_angst['angst'].values,frame_angst['angst_reg'].values))



## LinReg for EKEL (TSNE)

ekel_ratings = ekel['ekel_x']
ekel_predictors = np.asarray(ekel.loc[:,ekel.columns.values[np.where(ekel.columns.values == 'Ekel')[0][0]]:ekel.columns.values[-1]].values)

ekel_reg1 = label_extractor(ekel_predictors, ekel_ratings, stand_tsne_ekel.columns.values[1:],'Ekel',no_sublabels=15)
weights_per_label_ekel_tsne = weights_per_label
#Eye validity_check (see above)

ekel__reg_tsne = man.compute_single_column(ocw['ekel'],ekel_reg1)
frame_ekel = validation_preparation(db, ekel__reg_tsne[1], ekel_col_single[1], col_name='ekel_reg')

print(np.corrcoef(frame_ekel['ekel'], frame_ekel['ekel_reg single']))
print(np.corrcoef(frame_ekel['ekel'].values,frame_ekel['ekel_reg'].values))



## LinReg for FREUDE(TSNE)

freude_ratings = freude['freude_x']
freude_predictors = np.asarray(freude.loc[:,freude.columns.values[np.where(freude.columns.values == 'Freude')[0][0]]:freude.columns.values[-1]].values)

freude_reg1 = label_extractor(freude_predictors, freude_ratings, stand_tsne_freude.columns.values[1:], 'Freude',no_sublabels=15)
weights_per_label_freude_tsne = weights_per_label
#Eye validity_check (see above)

freude__reg_tsne = man.compute_single_column(ocw['freude'],freude_reg1)
frame_freude = validation_preparation(db, freude__reg_tsne[1], freude_col_single[1], col_name='freude_reg')

print(np.corrcoef(frame_freude['freude'], frame_freude['freude_reg single']))
print(np.corrcoef(frame_freude['freude'].values,frame_freude['freude_reg'].values))



## linReg for WUT (TSNE)

wut_ratings = wut['wut_x']
wut_predictors = np.asarray(wut.loc[:,wut.columns.values[np.where(wut.columns.values == 'Wut')[0][0]]:wut.columns.values[-1]].values)

wut_reg1 = label_extractor(wut_predictors, wut_ratings, stand_tsne_wut.columns.values[1:],'Wut',no_sublabels=16)
weights_per_label_wut_tsne = weights_per_label
#Eye validity_check (see above)

wut__reg_tsne = man.compute_single_column(ocw['wut'],wut_reg1)
frame_wut = validation_preparation(db, wut__reg_tsne[1], wut_col_single[1], col_name='wut_reg')

print(np.corrcoef(frame_wut['wut'], frame_wut['wut_reg single']))
print(np.corrcoef(frame_wut['wut'].values,frame_wut['wut_reg'].values))

## LinReg for TRAUER (TSNE)

trauer_ratings = trauer['trauer_x']
trauer_predictors = np.asarray(trauer.loc[:,trauer.columns.values[np.where(trauer.columns.values == 'Trauer')[0][0]]:trauer.columns.values[-1]].values)

trauer_reg1 = label_extractor(trauer_predictors, trauer_ratings, stand_tsne_trauer.columns.values[1:],'Trauer')
weights_per_label_trauer_tsne = weights_per_label
#Eye validity_check (see above)

trauer__reg_tsne = man.compute_single_column(ocw['trauer'],trauer_reg1)
frame_trauer = validation_preparation(db, trauer__reg_tsne[1], trauer_col_single[1], col_name='trauer_reg')

print(np.corrcoef(frame_trauer['trauer'], frame_trauer['trauer_reg single']))
print(np.corrcoef(frame_trauer['trauer'].values,frame_trauer['trauer_reg'].values))


### setting up a SentiArt table with DENN BAWL words ['sentiArt_table_DENN_BAWL.xlsx']

sentiArt = pd.DataFrame(index = db['up'].values, columns=['Angst single', 'Angst multi full', 'Angst multi tSNE', 'Trauer single',
                                                          'Trauer multi full', 'Trauer multi tSNE', 'Ekel single','Ekel multi full',
                                                         'Ekel multi tSNE', 'Freude single', 'Freude multi full', 'Freude multi tSNE',
                                                         'Wut single', 'Wut multi full', 'Wut multi tSNE'])

angst_s = man.compute_single_column(db['up'].values, {'Angst':'Angst'})
angst_mfull = man.compute_single_column(db['up'].values, angst_reg_full)
angst_mtSNE = man.compute_single_column(db['up'].values, angst_reg1)

ekel_s = man.compute_single_column(db['up'].values, {'Ekel':'Ekel'})
ekel_mfull = man.compute_single_column(db['up'].values, ekel_reg_full)
ekel_mtSNE = man.compute_single_column(db['up'].values, ekel_reg1)

freude_s = man.compute_single_column(db['up'].values, {'Freude':'Freude'})
freude_mfull = man.compute_single_column(db['up'].values, freude_reg_full)
freude_mtSNE = man.compute_single_column(db['up'].values, freude_reg1)

trauer_s = man.compute_single_column(db['up'].values, {'Trauer':'Trauer'})
trauer_mfull = man.compute_single_column(db['up'].values, trauer_reg_full)
trauer_mtSNE = man.compute_single_column(db['up'].values, trauer_reg1)

wut_s = man.compute_single_column(db['up'].values, {'Wut':'Wut'})
wut_mfull = man.compute_single_column(db['up'].values, wut_reg_full)
wut_mtSNE = man.compute_single_column(db['up'].values, wut_reg1)

column_data = [angst_s, angst_mfull, angst_mtSNE, trauer_s, trauer_mfull, trauer_mtSNE, ekel_s, ekel_mfull, ekel_mtSNE,
              freude_s, freude_mfull, freude_mtSNE, wut_s, wut_mfull, wut_mtSNE]

for x,y in zip(sentiArt.columns.values,column_data):
    sentiArt[x] = y[1]
    
sentiArt.to_excel('sentiArt_table_DENN_BAWL.xlsx')



### Analysis of correlations in relation to number of sublabels
#### ANGST

corrs_full = []
corrs_tsne = []

for i in range(1,50):
    label_dict_full = {'Angst': ['Angst'] + list(weights_per_label_angst.sort_values(ascending=False).index[:i])}
    col_full = man.compute_single_column(ocw['angst'], label_dict_full)
    frame_angst_full = validation_preparation(db, col_full[1], angst_col_single[1], col_name='angst_reg')
    corrs_full.append(np.corrcoef(frame_angst_full['angst_reg'].values, frame_angst_full['angst'].values)[0,1])


for i in range(1,50):
    label_dict_tsne = {'Angst': ['Angst'] + list(weights_per_label_angst_tsne.sort_values(ascending=False).index[:i])}
    col_tsne = man.compute_single_column(ocw['angst'], label_dict_tsne)
    frame_angst_tsne = validation_preparation(db, col_tsne[1], angst_col_single[1], col_name='angst_reg')
    corrs_tsne.append(np.corrcoef(frame_angst_tsne['angst_reg'].values, frame_angst_tsne['angst'].values)[0,1])

    
    
# ['corr per labelnumber angst.png']   
plt.plot(corrs_full, color='blue',label='full VSM')
plt.plot(corrs_tsne, color='blue',alpha=0.5, label='tsne VSM')
plt.xlabel('number of included sublabels', size=15)
plt.ylabel('correlation with target', size=15)
plt.title('Angst',size=15,loc='left')
plt.legend()


### Trauer

corrs_full = []
corrs_tsne = []

for i in range(1,50):
    label_dict_full = {'Trauer': ['Trauer'] + list(weights_per_label_trauer.sort_values(ascending=False).index[:i])}
    col_full = man.compute_single_column(ocw['trauer'], label_dict_full)
    frame_trauer_full = validation_preparation(db, col_full[1], trauer_col_single[1], col_name='trauer_reg')
    corrs_full.append(np.corrcoef(frame_trauer_full['trauer_reg'].values, frame_trauer_full['trauer'].values)[0,1])


for i in range(1,50):
    label_dict_tsne = {'Trauer': ['Trauer'] + list(weights_per_label_trauer_tsne.sort_values(ascending=False).index[:i])}
    col_tsne = man.compute_single_column(ocw['trauer'], label_dict_tsne)
    frame_trauer_tsne = validation_preparation(db, col_tsne[1], trauer_col_single[1], col_name='trauer_reg')
    corrs_tsne.append(np.corrcoef(frame_trauer_tsne['trauer_reg'].values, frame_trauer_tsne['trauer'].values)[0,1])

    
    
# ['corr per labelnumber trauer.png']    
plt.plot(corrs_full, color='orange',label='full VSM')
plt.plot(corrs_tsne, color='orange',alpha=0.5, label='tsne VSM')
plt.xlabel('number of included sublabels', size=15)
plt.ylabel('correlation with target', size=15)
plt.title('Trauer',size=15,loc='left')
plt.legend()

### Wut

corrs_full = []
corrs_tsne = []

for i in range(1,50):
    label_dict_full = {'Wut': ['Wut'] + list(weights_per_label_wut.sort_values(ascending=False).index[:i])}
    col_full = man.compute_single_column(ocw['wut'], label_dict_full)
    frame_wut_full = validation_preparation(db, col_full[1], wut_col_single[1], col_name='wut_reg')
    corrs_full.append(np.corrcoef(frame_wut_full['wut_reg'].values, frame_wut_full['wut'].values)[0,1])


for i in range(1,50):
    label_dict_tsne = {'Wut': ['Wut'] + list(weights_per_label_wut_tsne.sort_values(ascending=False).index[:i])}
    col_tsne = man.compute_single_column(ocw['wut'], label_dict_tsne)
    frame_wut_tsne = validation_preparation(db, col_tsne[1], wut_col_single[1], col_name='wut_reg')
    corrs_tsne.append(np.corrcoef(frame_wut_tsne['wut_reg'].values, frame_wut_tsne['wut'].values)[0,1])

    
    
# ['corr per labelnumber wut.png']   
plt.plot(corrs_full, color='red',label='full VSM')
plt.plot(corrs_tsne, color='red',alpha=0.5, label='tsne VSM')
plt.xlabel('number of included sublabels', size=15)
plt.ylabel('correlation with target', size=15)
plt.title('Wut',size=15,loc='left')
plt.legend()

### Freude

corrs_full = []
corrs_tsne = []

for i in range(1,50):
    label_dict_full = {'Freude': ['Freude'] + list(weights_per_label_freude.sort_values(ascending=False).index[:i])}
    col_full = man.compute_single_column(ocw['freude'], label_dict_full)
    frame_freude_full = validation_preparation(db, col_full[1], freude_col_single[1], col_name='freude_reg')
    corrs_full.append(np.corrcoef(frame_freude_full['freude_reg'].values, frame_freude_full['freude'].values)[0,1])


for i in range(1,50):
    label_dict_tsne = {'Freude': ['Freude'] + list(weights_per_label_freude_tsne.sort_values(ascending=False).index[:i])}
    col_tsne = man.compute_single_column(ocw['freude'], label_dict_tsne)
    frame_freude_tsne = validation_preparation(db, col_tsne[1], freude_col_single[1], col_name='freude_reg')
    corrs_tsne.append(np.corrcoef(frame_freude_tsne['freude_reg'].values, frame_freude_tsne['freude'].values)[0,1])

    
    
# ['corr per labelnumber freude.png']     
plt.plot(corrs_full, color='green',label='full VSM')
plt.plot(corrs_tsne, color='green',alpha=0.5, label='tsne VSM')
plt.xlabel('number of included sublabels', size=15)
plt.ylabel('correlation with target', size=15)
plt.title('Freude',size=15,loc='left')
plt.legend()


### Ekel

corrs_full = []
corrs_tsne = []

for i in range(1,50):
    label_dict_full = {'Ekel': ['Ekel'] + list(weights_per_label_ekel.sort_values(ascending=False).index[:i])}
    col_full = man.compute_single_column(ocw['ekel'], label_dict_full)
    frame_ekel_full = validation_preparation(db, col_full[1], ekel_col_single[1], col_name='ekel_reg')
    corrs_full.append(np.corrcoef(frame_ekel_full['ekel_reg'].values, frame_ekel_full['ekel'].values)[0,1])


for i in range(1,50):
    label_dict_tsne = {'Ekel': ['Ekel'] + list(weights_per_label_ekel_tsne.sort_values(ascending=False).index[:i])}
    col_tsne = man.compute_single_column(ocw['ekel'], label_dict_tsne)
    frame_ekel_tsne = validation_preparation(db, col_tsne[1], ekel_col_single[1], col_name='ekel_reg')
    corrs_tsne.append(np.corrcoef(frame_ekel_tsne['ekel_reg'].values, frame_ekel_tsne['ekel'].values)[0,1])

    
    
# ['corr per labelnumber ekel.png']     
plt.plot(corrs_full, color='purple',label='full VSM')
plt.plot(corrs_tsne, color='purple',alpha=0.5, label='tsne VSM')
plt.xlabel('number of included sublabels', size=15)
plt.ylabel('correlation with target', size=15)
plt.title('Ekel',size=15,loc='left')
plt.legend()



#### Analysis of cosine similarities and beta weights
# FULL model
angst_sims = [el[1] for el in label_frame['Angst']]
ekel_sims = [el[1] for el in label_frame['Ekel']]
trauer_sims = [el[1] for el in label_frame['Trauer']]
freude_sims = [el[1] for el in label_frame['Freude']]
wut_sims = [el[1] for el in label_frame['Wut']]

angst_corr = str(np.corrcoef(angst_sims, weights_per_label_angst.values)[0][1])[:6]
ekel_corr = str(np.corrcoef(ekel_sims, weights_per_label_ekel.values)[0][1])[:6]
trauer_corr = str(np.corrcoef(trauer_sims, weights_per_label_trauer.values)[0][1])[:6]
freude_corr = str(np.corrcoef(freude_sims, weights_per_label_freude.values)[0][1])[:6]
wut_corr = str(np.corrcoef(wut_sims, weights_per_label_wut.values)[0][1])[:6]

#['beta weights against cosine sim FULL model.png']
fig = plt.figure(figsize=(20,10))


plt.plot(angst_sims, weights_per_label_angst.values,color='blue', linewidth=2, 
         alpha=0.5, label='Angst: '+angst_corr)
plt.plot(ekel_sims, weights_per_label_ekel.values, color='purple',linewidth=2, 
         alpha=0.5, label='Ekel: '+ekel_corr)
plt.plot(trauer_sims, weights_per_label_trauer.values, color='orange',linewidth=2, 
         alpha=0.5, label='Trauer: '+trauer_corr)
plt.plot(freude_sims, weights_per_label_freude.values, color='green',linewidth=2, 
         alpha=0.5, label='Freude: '+freude_corr)
plt.plot(wut_sims, weights_per_label_wut.values, color='red',linewidth=2, 
         alpha=0.5, label='Wut: '+wut_corr)

plt.xlim(angst_sims[0], angst_sims[-1])
plt.xlabel('cosine similarity with main label', size=15)
plt.ylabel('beta weight',size=15)
plt.title('Full model',size=20, loc='left')
plt.legend()
plt.show()


#tSNE model
angst_tsne_corr = str(np.corrcoef(angst_sims, weights_per_label_angst_tsne.values)[0][1])[:5]
ekel_tsne_corr = str(np.corrcoef(ekel_sims, weights_per_label_ekel_tsne.values)[0][1])[:5]
trauer_tsne_corr = str(np.corrcoef(trauer_sims, weights_per_label_trauer_tsne.values)[0][1])[:5]
freude_tsne_corr = str(np.corrcoef(freude_sims, weights_per_label_freude_tsne.values)[0][1])[:6]
wut_tsne_corr = str(np.corrcoef(wut_sims, weights_per_label_wut_tsne.values)[0][1])[:5]

#['beta weights against cosine sim TSNE.png']
fig = plt.figure(figsize=(20,10))


plt.plot(angst_sims, weights_per_label_angst_tsne.values,color='blue', linewidth=2, 
         alpha=0.5, label='Angst: '+angst_tsne_corr)
plt.plot(ekel_sims, weights_per_label_ekel_tsne.values, color='purple',linewidth=2, 
         alpha=0.5, label='Ekel: ' + ekel_tsne_corr)
plt.plot(trauer_sims, weights_per_label_trauer_tsne.values, color='orange',linewidth=2, 
         alpha=0.5, label='Trauer: '+trauer_tsne_corr)
plt.plot(freude_sims, weights_per_label_freude_tsne.values, color='green',linewidth=2, 
         alpha=0.5, label='Freude: '+freude_tsne_corr)
plt.plot(wut_sims, weights_per_label_wut_tsne.values, color='red',linewidth=2, 
         alpha=0.5, label='Wut: '+wut_tsne_corr)

plt.xlim(angst_sims[0], angst_sims[-1])
plt.xlabel('cosine similarity with main label',size=15)
plt.ylabel('beta weight',size=15)
plt.title('tSNE model', size=20, loc='right')
plt.legend()
plt.show()



#### Multicolinearity and VIF

ekel_sublabel_df = pd.DataFrame(data=ekel_sublabel_matrix, index = range(ekel_sublabel_matrix.shape[0]), columns=just_labels['Ekel'])
freude_sublabel_df = pd.DataFrame(data=freude_sublabel_matrix, index = range(freude_sublabel_matrix.shape[0]), columns=just_labels['Freude'])
trauer_sublabel_df = pd.DataFrame(data = trauer_sublabel_matrix, index=range(trauer_sublabel_matrix.shape[0]), columns=just_labels['Trauer'])
angst_sublabel_df = pd.DataFrame(data=angst_sublabel_matrix, index=range(angst_sublabel_matrix.shape[0]), columns=just_labels['Angst'])
wut_sublabel_df = pd.DataFrame(data=wut_sublabel_matrix, index=range(wut_sublabel_matrix.shape[0]), columns=just_labels['Wut'])


ekel_sublabel_df_tsne = pd.DataFrame(data=ekel_predictors, index = range(ekel_predictors.shape[0]), columns=just_labels['Ekel'])
freude_sublabel_df_tsne = pd.DataFrame(data=freude_predictors, index = range(freude_predictors.shape[0]), columns=just_labels['Freude'])
trauer_sublabel_df_tsne = pd.DataFrame(data = trauer_predictors, index=range(trauer_predictors.shape[0]), columns=just_labels['Trauer'])
angst_sublabel_df_tsne = pd.DataFrame(data=angst_predictors, index=range(angst_predictors.shape[0]), columns=just_labels['Angst'])
wut_sublabel_df_tsne = pd.DataFrame(data=wut_predictors, index=range(wut_predictors.shape[0]), columns=just_labels['Wut'])


#Variance inflation factors for full and tSNE  model

from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(predictor_matrix):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = predictor_matrix.columns.values
    vif["VIF"] = [variance_inflation_factor(predictor_matrix.values, i) for i in range(predictor_matrix.shape[1])]

    return vif


angst_vif = calc_vif(angst_sublabel_df)
print(angst_vif.sort_values('VIF'))

angst_vif_tsne = calc_vif(angst_sublabel_df_tsne)
print(angst_vif_tsne.sort_values('VIF'))


ekel_vif = calc_vif(ekel_sublabel_df)
print(ekel_vif.sort_values('VIF'))

ekel_vif_tsne = calc_vif(ekel_sublabel_df_tsne)
print(ekel_vif_tsne.sort_values('VIF'))


trauer_vif = calc_vif(trauer_sublabel_df)
print(trauer_vif.sort_values('VIF'))

trauer_vif_tsne = calc_vif(trauer_sublabel_df_tsne)
print(trauer_vif_tsne.sort_values('VIF'))


freude_vif = calc_vif(freude_sublabel_df)
print(freude_vif.sort_values('VIF'))

freude_vif_tsne = calc_vif(freude_sublabel_df_tsne)
print(freude_vif_tsne.sort_values('VIF'))


wut_vif = calc_vif(wut_sublabel_df)
print(wut_vif.sort_values('VIF'))

wut_vif_tsne = calc_vif(wut_sublabel_df_tsne)
print(wut_vif_tsne.sort_values('VIF'))


### correlations with sublabels based on lowest VIFs
#Angst
angst_dict_vif = {'Angst': list(angst_vif.sort_values('VIF')['variables'][:15])}

angst__reg_vif = man.compute_single_column(ocw['angst'],angst_dict_vif)
frame_angst = validation_preparation(db, angst__reg_vif[1], angst_col_single[1], col_name='angst_reg')

print(np.corrcoef(frame_angst['angst'], frame_angst['angst_reg single']))
print(np.corrcoef(frame_angst['angst'].values,frame_angst['angst_reg'].values))

angst_dict_vif_tsne = {'Angst': list(angst_vif_tsne.sort_values('VIF')['variables'][:15])}

angst__reg_vif = man.compute_single_column(ocw['angst'],angst_dict_vif_tsne)
frame_angst = validation_preparation(db, angst__reg_vif[1], angst_col_single[1], col_name='angst_reg')

print(np.corrcoef(frame_angst['angst'], frame_angst['angst_reg single']))
print(np.corrcoef(frame_angst['angst'].values,frame_angst['angst_reg'].values))


#Ekel
ekel_dict_vif = {'Ekel': list(ekel_vif.sort_values('VIF')['variables'][:15])}

ekel__reg_vif = man.compute_single_column(ocw['ekel'],ekel_dict_vif)
frame_ekel = validation_preparation(db, ekel__reg_vif[1], ekel_col_single[1], col_name='ekel_reg')

print(np.corrcoef(frame_ekel['ekel'], frame_ekel['ekel_reg single']))
print(np.corrcoef(frame_ekel['ekel'].values,frame_ekel['ekel_reg'].values))

ekel_dict_vif_tsne = {'Ekel': list(ekel_vif_tsne.sort_values('VIF')['variables'][:15])}

ekel__reg_vif = man.compute_single_column(ocw['ekel'],ekel_dict_vif_tsne)
frame_ekel = validation_preparation(db, ekel__reg_vif[1], ekel_col_single[1], col_name='ekel_reg')

print(np.corrcoef(frame_ekel['ekel'], frame_ekel['ekel_reg single']))
print(np.corrcoef(frame_ekel['ekel'].values,frame_ekel['ekel_reg'].values))


#Trauer 
trauer_dict_vif = {'Trauer': list(trauer_vif.sort_values('VIF')['variables'][:15])}

trauer__reg_vif = man.compute_single_column(ocw['trauer'],trauer_dict_vif)
frame_trauer = validation_preparation(db, trauer__reg_vif[1], trauer_col_single[1], col_name='trauer_reg')

print(np.corrcoef(frame_trauer['trauer'], frame_trauer['trauer_reg single']))
print(np.corrcoef(frame_trauer['trauer'].values,frame_trauer['trauer_reg'].values))

trauer_dict_vif_tsne = {'Trauer': list(trauer_vif_tsne.sort_values('VIF')['variables'][:15])}

trauer__reg_vif = man.compute_single_column(ocw['trauer'],trauer_dict_vif_tsne)
frame_trauer = validation_preparation(db, trauer__reg_vif[1], trauer_col_single[1], col_name='trauer_reg')

print(np.corrcoef(frame_trauer['trauer'], frame_trauer['trauer_reg single']))
print(np.corrcoef(frame_trauer['trauer'].values,frame_trauer['trauer_reg'].values))


#Freude
freude_dict_vif = {'Freude': list(freude_vif.sort_values('VIF')['variables'][:15])}

freude__reg_vif = man.compute_single_column(ocw['freude'],freude_dict_vif)
frame_freude = validation_preparation(db, freude__reg_vif[1], freude_col_single[1], col_name='freude_reg')

print(np.corrcoef(frame_freude['freude'], frame_freude['freude_reg single']))
print(np.corrcoef(frame_freude['freude'].values,frame_freude['freude_reg'].values))

freude_dict_vif_tsne = {'Freude': list(freude_vif_tsne.sort_values('VIF')['variables'][:15])}

freude__reg_vif = man.compute_single_column(ocw['freude'],freude_dict_vif_tsne)
frame_freude = validation_preparation(db, freude__reg_vif[1], freude_col_single[1], col_name='freude_reg')

print(np.corrcoef(frame_freude['freude'], frame_freude['freude_reg single']))
print(np.corrcoef(frame_freude['freude'].values,frame_freude['freude_reg'].values))


#Wut
wut_dict_vif = {'Wut': list(wut_vif.sort_values('VIF')['variables'][:15])}

wut__reg_vif = man.compute_single_column(ocw['wut'],wut_dict_vif)
frame_wut = validation_preparation(db, wut__reg_vif[1], wut_col_single[1], col_name='wut_reg')

print(np.corrcoef(frame_wut['wut'], frame_wut['wut_reg single']))
print(np.corrcoef(frame_wut['wut'].values,frame_wut['wut_reg'].values))

wut_dict_vif_tsne = {'Wut': list(wut_vif_tsne.sort_values('VIF')['variables'][:15])}

wut__reg_vif = man.compute_single_column(ocw['wut'],wut_dict_vif_tsne)
frame_wut = validation_preparation(db, wut__reg_vif[1], wut_col_single[1], col_name='wut_reg')

print(np.corrcoef(frame_wut['wut'], frame_wut['wut_reg single']))
print(np.corrcoef(frame_wut['wut'].values,frame_wut['wut_reg'].values))



#### corr(VIF, cosine_sim with main label) analysis
#Angst
angst_frame_vif_cos = pd.DataFrame(index = range(200), columns=['word','Cosine Similarity','VIF full', 'VIF tsne'])
                                   
angst_frame_vif_cos['word'] = [el[0] for el in label_frame['Angst'].values]
angst_frame_vif_cos['Cosine Similarity'] = [el[1] for el in label_frame['Angst'].values]
angst_frame_vif_cos['VIF full'] = angst_vif['VIF']
angst_frame_vif_cos['VIF tSNE'] = angst_vif_tsne['VIF']

angst_frame_vif_cos.corr()      

#Ekel
ekel_frame_vif_cos = pd.DataFrame(index = range(200), columns=['word','Cosine Similarity','VIF full', 'VIF tsne'])
                                   
ekel_frame_vif_cos['word'] = [el[0] for el in label_frame['Ekel'].values]
ekel_frame_vif_cos['Cosine Similarity'] = [el[1] for el in label_frame['Ekel'].values]
ekel_frame_vif_cos['VIF full'] = ekel_vif['VIF']
ekel_frame_vif_cos['VIF tSNE'] = ekel_vif_tsne['VIF']

ekel_frame_vif_cos.corr()

#Trauer
trauer_frame_vif_cos = pd.DataFrame(index = range(200), columns=['word','Cosine Similarity','VIF full', 'VIF tsne'])
                                   
trauer_frame_vif_cos['word'] = [el[0] for el in label_frame['Trauer'].values]
trauer_frame_vif_cos['Cosine Similarity'] = [el[1] for el in label_frame['Trauer'].values]
trauer_frame_vif_cos['VIF full'] = trauer_vif['VIF']
trauer_frame_vif_cos['VIF tSNE'] = trauer_vif_tsne['VIF']

trauer_frame_vif_cos.corr()

#Freude
freude_frame_vif_cos = pd.DataFrame(index = range(200), columns=['word','Cosine Similarity','VIF full', 'VIF tsne'])
                                   
freude_frame_vif_cos['word'] = [el[0] for el in label_frame['Freude'].values]
freude_frame_vif_cos['Cosine Similarity'] = [el[1] for el in label_frame['Freude'].values]
freude_frame_vif_cos['VIF full'] = freude_vif['VIF']
freude_frame_vif_cos['VIF tSNE'] = freude_vif_tsne['VIF']

freude_frame_vif_cos.corr()

#Wut
wut_frame_vif_cos = pd.DataFrame(index = range(200), columns=['word','Cosine Similarity','VIF full', 'VIF tsne'])
                                   
wut_frame_vif_cos['word'] = [el[0] for el in label_frame['Wut'].values]
wut_frame_vif_cos['Cosine Similarity'] = [el[1] for el in label_frame['Wut'].values]
wut_frame_vif_cos['VIF full'] = wut_vif['VIF']
wut_frame_vif_cos['VIF tSNE'] = wut_vif_tsne['VIF']

wut_frame_vif_cos.corr()



################################## CLASSIFICATION (single_label vs. multilabel_full vs. multilabel_tsne)   ##############################################

single_cols = [angst_col_single, ekel_col_single, trauer_col_single, freude_col_single, wut_col_single]
multi_cols_full = [angst__reg_full, ekel__reg_full, trauer__reg_full, freude__reg_full, wut__reg_full]
multi_cols_tsne = [angst__reg_tsne, ekel__reg_tsne, trauer__reg_tsne, freude__reg_tsne, wut__reg_tsne]


values_single = pd.DataFrame(index = shared, columns = ['angst', 'ekel', 'trauer', 'freude', 'wut'])
                                    
values_multi_full = pd.DataFrame(index = shared, columns = ['angst', 'ekel', 'trauer', 'freude', 'wut'])

values_multi_tsne = pd.DataFrame(index = shared, columns = ['angst', 'ekel', 'trauer', 'freude', 'wut'])


for col, comp in zip(values_single, single_cols):
    values_single[col] = [comp[1][index] for index in comp[1].index if index in shared]

for m_col, m_comp in zip(values_multi_full, multi_cols_full):
    values_multi_full[m_col] = [m_comp[1][index] for index in m_comp[1].index if index in shared]
    
for t_col, t_comp in zip(values_multi_tsne, multi_cols_tsne):
     values_multi_tsne[t_col] = [t_comp[1][index] for index in t_comp[1].index if index in shared]



# Classifier target

for row in db.index:
    db.loc[row,('MAXLABEL_VALUE_target')] = max(db.loc[row]['freude'],db.loc[row]['angst'],db.loc[row]['wut'],
                                  db.loc[row]['ekel'], db.loc[row]['trauer'])
    

    
value_frame = pd.DataFrame(index = db.index, columns=[db.columns.values[18],db.columns.values[20],db.columns.values[22],db.columns.values[24],
                                                     db.columns.values[26]], data = np.transpose(np.asarray([db['freude'].values, 
                                                                                                             db['wut'].values, db['trauer'].values,
                                                                                                                          db['angst'].values, db['ekel'].values])))
db['MAXLABEL_target'] = value_frame.idxmax(axis=1, skipna=True)

# populate a df with classification-relevant information
class_frame = pd.DataFrame(index = shared, columns = ['MAXLABEL_predictor_single', 'MAXLABEL_predictor_multi_full','MAXLABEL_predictor_multi_tsne', 'True Category'])


class_frame['MAXLABEL_predictor_single'] = values_single.idxmax(axis=1, skipna=True)
class_frame['MAXLABEL_predictor_multi_full'] = values_multi_full.idxmax(axis=1, skipna=True)
class_frame['MAXLABEL_predictor_multi_tsne'] = values_multi_tsne.idxmax(axis=1, skipna=True)
class_frame['True Category'] = [db.loc[i]['MAXLABEL_target'] for i in db.index if db.loc[i]['up'] in shared]


from sklearn.metrics import confusion_matrix


classification_singlelabel = confusion_matrix(class_frame['True Category'].values, 
                                              class_frame['MAXLABEL_predictor_single'].values)


classification_multilabel_full = confusion_matrix(class_frame['True Category'].values,
                                             class_frame['MAXLABEL_predictor_multi_full'].values)

classification_multilabel_tsne = confusion_matrix(class_frame['True Category'].values,
                                             class_frame['MAXLABEL_predictor_multi_tsne'].values)



import itertools

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





# single label confusion matrix ['con mat DENN BAWL single.png']
plot_confusion_matrix(classification_singlelabel, classes=list(set(class_frame['True Category'].values)), 
                      normalize=True, title= 'Singlelabel')

print(accuracy_score(class_frame['True Category'].values, 
                                              class_frame['MAXLABEL_predictor_single'].values))
print(f1_score(class_frame['True Category'].values, 
                                              class_frame['MAXLABEL_predictor_single'].values, average='macro'))

# multilabel full confusion matrix ['con mat DENN BAWL multi full']
plot_confusion_matrix(classification_multilabel_full, classes=list(set(class_frame['True Category'].values)), 
                      normalize=True, title='Multilabel full VSM')

print(accuracy_score(class_frame['True Category'].values, 
                                              class_frame['MAXLABEL_predictor_multi_full'].values))

print(f1_score(class_frame['True Category'].values, 
                                              class_frame['MAXLABEL_predictor_multi_full'].values, average='macro'))

# multilabel tsne confusion matrix ['con mat DENN BAWL multi tSNE']
plot_confusion_matrix(classification_multilabel_tsne, classes=list(set(class_frame['True Category'].values)), 
                      normalize=True, title='Multilabel tSNE')
print(accuracy_score(class_frame['True Category'].values, 
                                              class_frame['MAXLABEL_predictor_multi_tsne'].values))
print(f1_score(class_frame['True Category'].values, 
                                              class_frame['MAXLABEL_predictor_multi_tsne'].values, average='macro'))

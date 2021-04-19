'''
Preprocessing of Harry_Potter chapter based on the Output of the
treetagger (treetagger_harry_output.txt).

The aim is to extract all nouns from the text and group the according to the
sentences in which they appear.

Every sentence is then supplemented with its respective human ratings for
'ANGST' und 'FREUDE' and a rated category (H('Happiness'),
F('Fear') or N('Neutral')).
'''

import pandas as pd
import nltk
import os
import numpy as np
import gensim
import re


from gensim.models import KeyedVectors
model = ''  # insert model directory

vsm = KeyedVectors.load_word2vec_format(model)


working_dir ='' # insert working directory where relevant files are stored
os.chdir(working_dir)

with open('treetagger_harry_output.txt', encoding='utf-8') as f:
    lines = f.readlines()                       
    ## lines is a list of strings, each str representing one line in the file
    

with open('120HPde_ratings.txt', encoding='utf-8') as g:
    harry_lines = g.readlines()    

harry_df = pd.DataFrame(index=range(1,121), columns=['nouns', 'Category', 
                                                     'Mean_Fear_RATING', 'Mean_happiness_RATING'])


harry_df['nouns'] = ''


## filter out lines with sentence ratings
limiter_indices = []


for line in lines:
    for letter in line:
        if letter.isnumeric():
            limiter_indices.append(lines.index(line))

limiters = list(set(limiter_indices))



for limiter in limiters:
    if '<unknown>' not in lines[limiter]:
        limiters.remove(limiter)


limiters.sort()



## filter nouns
low_lim = 3

nouns_and_index = []

for i in range(low_lim, len(lines)):
    if 'NN' in lines[i]:
        if i == 422:
            nouns_and_index.append(list(('Kopfes',i)))  # exception based on eye-inspection of (lines)
        else:
            nouns_and_index.append(list((lines[i].split('\t')[0], i)))
            
for el in nouns_and_index:
    if el[1] in limiters and el[0][0].isupper() == False:
        print(el)
        nouns_and_index.remove(el)
        
for el in nouns_and_index:
    if any(char.isdigit() for char in el[0]):
        el[0] = el[0].split('.')[0]



## clean nouns
# check for false negatives
for line in lines:
    if (line[0].isupper() or line[1].isupper()) and 'NN' not in line:
        print(line)
        
# check for false positives 
for element in nouns_and_index:
    if element[0] not in vsm.vocab:
        print(element)
        nouns_and_index.remove(element)
        

# EYE VALIDITY CHECK

nouns_and_index.remove(['Nein', 1136])
nouns_and_index.remove(['Jetzt', 1447])
nouns_and_index.remove(['Sein', 1817])
nouns_and_index.remove(['Dicke', 2181])
nouns_and_index.remove(['Großen', 2202])
nouns_and_index.remove(['Es', 2391])
nouns_and_index.remove(['Ja', 3606])
nouns_and_index.remove(['Cross', 3944])
nouns_and_index.remove(['Tschau', 4044])
nouns_and_index.remove(['Cross', 4092])
nouns_and_index.remove(['Sprechende', 4311])
nouns_and_index.remove(['Danke', 5581])
nouns_and_index.remove(['Es', 5638])
nouns_and_index.remove(['Runenwörterbüchern', 4475])




##set up harry_df

lower_limit = 0
for lim in limiters:
    for element in nouns_and_index:
        if lower_limit < element[1] < lim:
            harry_df.iloc[limiters.index(lim)]['nouns'] += element[0] + ' '
        else:
            continue
    
    lower_limit = lim


for i in harry_df.index.values:
    harry_df.loc[i]['nouns'] = list(harry_df.loc[i]['nouns'].split(' ')[:-1])


#filter out lines with problematic format for information extraction

problematic_lines = dict()
prob = []

for line in harry_lines[1:]:
    harry_df.loc[harry_lines.index(line)]['Category'] = re.search(',[FHN],', line).group()[1]
    try:
        harry_df.loc[harry_lines.index(line)]['Mean_Fear_RATING'] = float(
            re.findall('-?\d,\d+', line)[1].replace(',','.'))
        harry_df.loc[harry_lines.index(line)]['Mean_happiness_RATING'] = float(
            re.findall('-?\d,\d+', line)[2].replace(',','.'))
    except:
        problematic_lines[harry_lines.index(line)] = line.rstrip()[line.index(re.search(',[FHN],', line).group()):]
        prob.append(line.rstrip()[line.index(re.search(',[FHN],', line).group()):])

for i in problematic_lines.keys():
    for j in range(len(problematic_lines[i])):
        if len(problematic_lines[i]) - 1 > j > 0:
            if problematic_lines[i][j] == ',' and (problematic_lines[i][j-1].isdigit() and problematic_lines[i][j+1].isdigit()):
                s = list(problematic_lines[i])
                s[j] = '.'
                problematic_lines[i] = "".join(s)
    problematic_lines[i] = problematic_lines[i].replace('"','').split(',')



for key in problematic_lines.keys():
    harry_df.loc[key]['Mean_Fear_RATING'] = float(problematic_lines[key][-2])
    harry_df.loc[key]['Mean_happiness_RATING'] = float(problematic_lines[key][-1])


## save frame to excel file
#harry_df.to_excel('harry_120_preprocessed.xlsx')

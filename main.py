import csv
import pandas as pd
import os
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


target_file = 'reviews.csv'

test_data = []
train_data = []


def convert_file():
    if os.path.isfile(target_file):
        return

    with open('reviews.tsv','rb') as fin:
        cr = csv.reader(fin, delimiter='\t')
        filecontents = [line for line in cr]

    with open(target_file,'wb') as fou:
        cw = csv.writer(fou, quoting=csv.QUOTE_ALL,escapechar='\\')
        cw.writerows(filecontents)

def divide_by_rankings():
    colnames = ['rating', 'text']
    contents = pd.read_csv(target_file, names=colnames, header=None)

    contents['rating'] = contents['rating'].astype(int)

    train_data, test_data = train_test_split(contents, test_size=0.2)

    negative_rankings = train_data[train_data['rating'] < 3]
    neutral_documents = train_data[train_data['rating'] == 3]
    positive_rankings = train_data[train_data['rating'] > 3]

    return negative_rankings, neutral_documents, positive_rankings

def tokenize(docs, file_name):

    tokens = docs['text'].values
    tokens = ' '.join(tokens).split()
    tokens = set(tokens)
    tokens = [str(x.lower()) for x in tokens if x not in ENGLISH_STOP_WORDS]

    with open(file_name, 'w') as the_file:
        the_file.write('\n'.join(tokens))

        
def main():
    convert_file()
    negative_docs, neutral_docs, positive_docs = divide_by_rankings()

    tokenize(negative_docs, 'negative.txt')
    tokenize(neutral_docs, 'neutral.txt')
    tokenize(positive_docs, 'positive.txt')





if __name__ == "__main__":
    main()
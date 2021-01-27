import nltk
import numpy as np
import pandas as pd
import pickle
import re
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from sklearn_crfsuite import scorers
from .utils import *

class NER_CRF:
    def __init__(self,
                algorithm='l2sgd',
                c2=0.1,
                max_iterations=1000,
                all_possible_transitions=True):

        self.crf = CRF(
            algorithm = algorithm,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=all_possible_transitions)
    
    def train(self, input="Dataset/ETCSL_ORACC_NER.csv", 
             output="Saved_Models/NER/NER_CRF.pkl", 
             test_size = 0.05, 
             random_state=7):

        df=pd.read_csv(input)
        tagged_sentence=Preparing_tagged_data(df)
        df=df[['ID','FORM','XPOSTAG']]
        #printing details
        printing_details(tagged_sentence)
        
        train_set, test_set = train_test_split(tagged_sentence,test_size=test_size,random_state=random_state)
        
        X_train,y_train=prepareData(tagged_sentence)
        X_test,y_test=prepareData(test_set)
        
        self.crf.fit(self.X_train, self.y_train)
        print(self.crf)
        
        print("Saving Model .....")
        # Save the Model to file in the current working directory
        Pkl_Filename = output
        with open(Pkl_Filename, 'wb') as file:
            pickle.dump(self.crf, file)
            
        print("Model Saved at "+ Pkl_Filename)
        print()    
        print("Checking the Algoritham's Performance \n")
        self.test(X_train,y_train,X_test,y_test) 

    def test(self, X_train,y_train,X_test,y_test):

        y_pred=self.crf.predict(X_test)
        y_pred_train=self.crf.predict(X_train)
        print("training accuracy is %f \n" % metrics.flat_f1_score(y_train, y_pred_train,average='weighted',labels=self.crf.classes_))
        print("test accuracy is %f \n" % metrics.flat_f1_score(y_test, y_pred,average='weighted',labels=self.crf.classes_))
        print("Test classification report is \n")
        print(metrics.flat_classification_report(y_test, y_pred, labels=self.crf.classes_, digits=3))

    def predict(self, input="Dataset/sumerian_demo.txt", 
                saved="Saved_Models/NER/NER_CRF.pkl",
                output='Output/NER_CRF.txt'):

        Monolingual_sumerian= Openfile(input)

        with open(saved, 'rb') as file:  
            crf = pickle.load(file)

        Processed_sumerian_monolingual=prepare_test_Data(Monolingual_sumerian)

        Prediction=crf.predict(Processed_sumerian_monolingual)

        POS_list=POSLIST(Monolingual_sumerian,Prediction)

        print("Saving_file "+output)
        Savefile(output,Monolingual_sumerian,POS_list)
    












#IF WE WANT TO USE CLASS 
#pipeline=OPEN('ATF_OUTPUT/pipeline.txt')
#POS=POS_tag(Pipeline)
#savefile('ATF_OUTPUT/pipeline1.txt',POS)
#NER=NER_tag(Pipeline)
#savefile('ATF_OUTPUT/pipeline2.txt',NER)
#translations=Translation_tag(Pipeline)
#savefile('ATF_OUTPUT/pipeline3.txt',translations)
#input = default="ATF_INPUT/demo.atf"
#output= ATF_OUTPUT/
#ner_path='NER_Models/'+args.ner+'/prediction.py'
#'python3 {ner_path} -i {output_dir}pipeline.txt -o {output_dir}ner_pipeline.txt'
#python NER_Models/NER_CRF/prediction.py -i ATF_OUTPUT/pipeline.txt -o ATF_OUTPUT/ner_pipeline.txt
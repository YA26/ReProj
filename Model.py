import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.utils import shuffle
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
import tensorflow as tf
from collections import defaultdict
from nltk import RegexpTokenizer
from sklearn.preprocessing import OneHotEncoder
import operator
import re
import pickle

class ModelBuilder:
    
    def __init__(self, Word_2_Vec_path):
        self.__toknizer         = RegexpTokenizer(r'''\w'|\w+|[^\w\s]''')
        self.__Word2_vec_model  = KeyedVectors.load_word2vec_format(Word_2_Vec_path, binary=True)  
        self.__one_hot_encoder  = OneHotEncoder()
              
    def get_word_2_vec_model(self):
        return self.__Word2_vec_model
    
    def get_tokenizer(self):
        return self.__toknizer
    
    def get_one_hot_encoder(self):
        return self.__one_hot_encoder
    
    def sent_tokenizer(self, string):
        '''
        Self_made sentence tokenizer because sent_tokenizer of nltk doesn't split properly.
        Especially if there's no space after a full stop or other punctuations
        '''
        return re.split("[?.!]+", re.sub("[?.!]$", "", string))

    def get_rid_of_unwanted_chars(self, corpus):
        '''
        With this function, we write a regex to remove unwanted punctuations, email adresses and other elements
        that are useless to our tasks and our model
        '''
        return " ".join(re.findall("[\D.]+", " ".join(re.findall("[\w.]+", corpus))))
    
    def sentence_to_vector(self, X_not_encoded):
        '''
        Function to implement sentence embedding 
        '''
        #We use Word2Vec embedding, if there is a vector for a given word, we use its vector, otherwise we discard it.
        X_encoded=[] 
        for sentence in X_not_encoded:
            sent_vector=0
            for word in self.__toknizer.tokenize(sentence.lower()):
                try:
                    sent_vector+=self.__Word2_vec_model.get_vector(word)
                except KeyError:
                    pass
            X_encoded.append(sent_vector)
        X_encoded=np.array(X_encoded)
        return X_encoded
                  
    def build_training_set(self, contents_by_categories):
        '''
        Encoding, shuffling and gathering everything up 
        '''
        #Creating not_encoded X_train and y_train datasets from contents_by_categories
        X_train_not_encoded=[]
        y_train_not_encoded=[]
        for index, corpora in contents_by_categories.items():
            for corpus in corpora:
                X_train_not_encoded.append(corpus[0])
                y_train_not_encoded.append(index)
        
        X_train_not_encoded=np.array(X_train_not_encoded)
        y_train_not_encoded=np.array(y_train_not_encoded)
        
        #Encoding our training_set contents and labels 
        X_train_encoded=self.sentence_to_vector(X_train_not_encoded)
        
        #Encoding our training_set labels 
        y_train_encoded=[] 
        y_train_not_encoded=y_train_not_encoded.reshape((-1,1))
        self.__one_hot_encoder.fit(y_train_not_encoded)
        y_train_encoded = self.__one_hot_encoder.transform(y_train_not_encoded).toarray()
        
        #Shuffling training_set 
        X_and_y=np.concatenate((X_train_encoded, y_train_encoded), axis=1)
        X_and_y=shuffle(X_and_y)      
        X_train_encoded=X_and_y[:,:X_train_encoded.shape[1]]
        y_train_encoded=X_and_y[:,X_train_encoded.shape[1]:]
        return X_train_encoded, y_train_encoded
       
    def neural_network(self, X, y, path, epoch, batch_size, lr, validation_split):
        '''
        This functions implements a neural network with some dropout layers and cross validation to avoid overfitting. 
        Here we start with an input vector of 500 dimensions. Every dense layer maps the former input to a reduced output to allow
        every single perceptron to have a better classification of all the inputs(sentences) in our dataset
        '''
        model = tf.keras.Sequential()
        model.add(layers.Dense(units=X.shape[1] , input_dim=X.shape[1]))
        model.add(Dropout(0.2))
        model.add(layers.Dense(units=400, activation='relu'))
        model.add(Dropout(0.2))
        model.add(layers.Dense(units=200, activation='relu'))
        model.add(Dropout(0.2))
        model.add(layers.Dense(units=100, activation='relu'))
        model.add(Dropout(0.2))
        model.add(layers.Dense(units=50, activation='relu'))
        model.add(Dropout(0.2))
        model.add(layers.Dense(units=y.shape[1], activation='softmax'))
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=[tf.keras.metrics.Precision()], lr=lr, validation_split=validation_split) 
        model.fit(X, y, epochs=epoch, batch_size=batch_size)
        model.save(path)
        print("Saved model to "+str(path))
        return model
      
    def cosine_similarity(self, word, tf_idf_scores):
        '''
        This functions helps us determine how valuable a word is to our targets variables.
        Every word from every single sentence must bring an added value that will make our predictions easier.
        We then have to compute the cosine similarity of every word with a given word in tf_idf[category] and discard words that are not similar with any words in tf_idf[category]
        '''
        try:
            word_vector=self.__Word2_vec_model.get_vector(word)
            cosines_dict=defaultdict(int) 
            tf_idf_words=tf_idf_scores.keys()  
            for tf_idf_word in tf_idf_words:  
                tf_idf_word_vector=self.__Word2_vec_model.get_vector(tf_idf_word)
                cos=np.dot(word_vector, tf_idf_word_vector)/(np.linalg.norm(word_vector)*np.linalg.norm(tf_idf_word_vector))   
                if cos<0:
                    cos=0
                cosines_dict[tf_idf_word]=cos
            td_idf_max_value=np.max(list(cosines_dict.values()))
            tf_idf_word_of_max_value=max(cosines_dict.items(), key=operator.itemgetter(1))[0]  
            #if cosine is greater than 0.60, we keep the word
            if td_idf_max_value > 0.60:
                return tf_idf_scores[tf_idf_word_of_max_value]
        except KeyError:
            return 0
        return 0

    def shrink_sentence(self, context_word, sentence, window):
        '''
        shrinks a sentence by keeping words that are in the vicinity(window) of the context_word
        '''
        sent_index=sentence.index(context_word)
        sentence_1=" ".join(sentence[:sent_index].split()[-window:])
        sentence_2=" ".join(sentence[sent_index+len(context_word):].split()[:window])
        new_sentence="{} {} {}".format(sentence_1, context_word, sentence_2)
        return new_sentence
    
    def get_meaningful_sentences_only_with_label(self, tf_idf_dict, X_test, y_test):
        '''
        This functions helps us clean sentences of words that are irrelevant
        It has to be applied on a test set
        
        1-We first get sentences that are relevant. If a sentence has the potential to be classified in any of our category, we keep it. 
        So we have to see how relevant its words are for any potential category
        
        2- We then shrink sentences to reduce noise

        3- We keep the remaining sentence
        '''
        X=[]
        y=[]
        iterator=1
        for corpus, categories in zip(X_test, y_test):
            corpus_=""
            print("RELEVANT SENTENCES OF CORPUS {} ARE BEING RETRIEVED".format(iterator))
            clean_corpus=self.get_rid_of_unwanted_chars(corpus)
            for sentence in self.sent_tokenizer(clean_corpus.lower()):
                sentences_shrunk=[]
                sentence_score=0
                #We try to see if perhaps a sentence is relevant to any category
                for word in self.__toknizer.tokenize(sentence):
                    #We have to consider every word in order to see how they are related to words in tf_idf[category]
                    score=0 
                    for label, tf_idf_scores in tf_idf_dict.items():
                        try:
                            score=tf_idf_scores[word]
                            sentence_score+=score
                        except KeyError:
                            score=self.cosine_similarity(word, tf_idf_scores)
                            sentence_score+=score
                        if score>0:
                            sentences_shrunk.append("{}. ".format(self.shrink_sentence(word, sentence, 3)))
                if sentence_score>0:
                    corpus_+=" ".join(list(dict.fromkeys(sentences_shrunk)))
            if len(corpus_.strip())>0:
                X.append(corpus_)
                y.append(categories)
            iterator+=1
        return np.array(X), np.array(y)
    
    def get_only_meaningful_sentences_without_label(self, tf_idf_dict, X):
        '''
        This functions helps us clean sentences of words that are irrelevant
        It has to be used for a new observation
        '''
        X_new=[]
        iterator=1
        for corpus in X:
            corpus_=""
            print("RELEVANT SENTENCES OF CORPUS {} ARE BEING RETRIEVED".format(iterator))
            clean_corpus=self.get_rid_of_unwanted_chars(corpus)
            for sentence in self.sent_tokenizer(clean_corpus.lower()):
                sentences_shrunk=[]
                sentence_score=0
                for word in self.__toknizer.tokenize(sentence):
                    score=0 
                    for label, tf_idf_scores in tf_idf_dict.items():
                        try:
                            score=tf_idf_scores[word]
                            sentence_score+=score
                        except KeyError:
                            score=self.cosine_similarity(word, tf_idf_scores)
                            sentence_score+=score
                        if score>0:
                            sentences_shrunk.append("{}. ".format(self.shrink_sentence(word, sentence, 3)))       
                if sentence_score>0:
                    corpus_+=" ".join(list(dict.fromkeys(sentences_shrunk)))
            if len(corpus_.strip())>0:
                X_new.append(corpus_)
            iterator+=1
        return np.array(X_new)
              
    def predict(self, X, stat_model, categories):
        '''
        We predict a label for every sentence in a corpus and we average all the predictions:
        1- We first compute the average scores of all sentences in the corpus by:
            a- tokenizing every sentence and making predictions for every single one of them
            b- averaging all those predictions. We then get a prediction dataframe for all sentences

        2- We then compute scores for the corpus as a whole by giving it to our neural network. We get a 
        prediction dataframe for the corpus

        3- We average the two dataframes and we get our final predictions(the best of both worlds)
        '''

        # max_predictions_list to store our final predictions
        max_predictions=[]
        for counter, corpus in enumerate(X):
            print("PREDICTIONS FOR CORPUS "+str(counter+1)+" ARE BEING PRODUCED")
            #df_sentences_prediction to get average prediction in a sentence by sentence way
            df_sentences_prediction=pd.DataFrame(np.zeros(len(categories)), index=categories, columns=["Predictions"])
            #df_corpus_prediction to get prediction for the whole corpus
            df_corpus_prediction=pd.DataFrame(np.zeros(len(categories)), index=categories, columns=["Predictions"])
            number_of_sentences=0
            corpus_vector=0
            for sentence in self.sent_tokenizer(corpus):
                sentence_vector=0
                for word in self.__toknizer.tokenize(sentence):
                    try: 
                        sentence_vector+=self.__Word2_vec_model.get_vector(word.lower())
                    except KeyError:
                        pass
                corpus_vector+=sentence_vector
                if type(sentence_vector)!=int:
                    number_of_sentences+=1
                    prediction_sentence_vector=stat_model.predict(sentence_vector.reshape(1,-1)).T  
                    #We average all the predicitions   
                    df_sentences_prediction+=pd.DataFrame(prediction_sentence_vector*100, index=categories, columns=["Predictions"])

            if df_sentences_prediction is not None:
                prediction_corpus_vector=stat_model.predict(corpus_vector.reshape(1,-1)).T 
                df_corpus_prediction=pd.DataFrame(prediction_corpus_vector*100, index=categories, columns=["Predictions"])        
                df_sentences_prediction/=number_of_sentences
                #We get the best of both worlds
                df_final_prediction=(df_corpus_prediction+df_sentences_prediction)/2
                max_predictions.append([df_final_prediction.idxmax()[0], df_final_prediction.max()[0]])
        return np.array(max_predictions)
    
    def save_variables(self, path, variable):
        '''
        Save variables with pickle
        '''
        with open(path, "wb") as file:
            pickle.dump(variable, file)
    
    def load_variables(self, path):
        '''
        Load variables with pickle
        '''
        loaded_variable=None
        with open(path, "rb") as file:
            loaded_variable=pickle.load(file)
        return loaded_variable
    
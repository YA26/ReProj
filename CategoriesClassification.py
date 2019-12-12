from DataPreprocessing import DataPreProcessing
from  Model import ModelBuilder
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report


"""
############################################
############# MAIN OBJECTS #################
############################################
"""

data_preprocessing  =   DataPreProcessing(word2vec_path="./Word2Vec/frWac_non_lem_no_postag_no_phrase_500_skip_cut200.bin")
build_model         =   ModelBuilder(word2vec_path="./Word2Vec/frWac_non_lem_no_postag_no_phrase_500_skip_cut200.bin") 

"""
############################################
############# SCRAPING DATA ################
############################################
"""
#query_and_labels=pd.read_csv("./data/scraping/query_and_labels.csv", encoding="ISO-8859-1")
#data_preprocessing.scrape_google_search(query_and_labels, "./data/scraping/scraped_data_3.csv","C:/Program Files/Mozilla Firefox/firefox.exe", "C:/Users/BB/Desktop/Vilogia/gecko_driver/geckodriver.exe", 1)
data_scraped=pd.read_csv("./data/scraping/scraped_data.csv")

"""
############################################
######## DATA PREPROCESSING ################
############################################
"""


original_dataset                                    =   pd.read_csv("./data/train_v2.csv", sep=";", encoding="ISO-8859-1")
concat_dataset                                      =   pd.concat([original_dataset, data_scraped])
X_train, X_test, y_train, y_test                    =   data_preprocessing.train_test_separator(content=concat_dataset["content"].values, labels=concat_dataset["label"].values, test_size=0.2)
contents_by_categories                              =   data_preprocessing.group_contents_by_label(X=X_train, y=y_train)
contents_by_categories_cleansed_up, unique_words    =   data_preprocessing.clean_up_files(contents_by_categories)

#data_preprocessing.hist_corpus_per_label(contents_by_categories_cleansed_up, "./graphs/contents_by_categories")
#term frequency and inverse document frequency dictionaries 
tf_dict, idf_dict       =   data_preprocessing.tf_and_idf(contents_by_categories_cleansed_up)
tf_idf_dict             =   data_preprocessing.tf_idf_scores(tf_dict, idf_dict, tf_idf_z_score_threshold=2.8, cos_z_score_threshold=0, verbose=True)
sentences_scored_dict   =   data_preprocessing.score_sentences(contents_by_categories_cleansed_up, tf_idf_dict)
  
"""
############################################
############## TRAINING ####################
############################################
"""
X_train_encoded, y_train_encoded    =   build_model.build_training_set(sentences_scored_dict)
categories_classification_model     =   build_model.neural_network(X=X_train_encoded, 
                                                                   y=y_train_encoded, 
                                                                   path="./ML_models/rental_incidents_classification_model_2.h5", 
                                                                   epoch=50, 
                                                                   batch_size=200,
                                                                   lr=0.001,
                                                                   validation_split=0.1)

"""
############################################
##############SAVING VARIABLES##############
############################################
"""
"""
one_hot_encoder=build_model.get_one_hot_encoder()
build_model.save_variables(path="./saved_variables/X_train.pickle", variable=X_train)
build_model.save_variables(path="./saved_variables/y_train.pickle", variable=y_train)
build_model.save_variables(path="./saved_variables/X_test.pickle", variable=X_test)
build_model.save_variables(path="./saved_variables/y_test.pickle", variable=y_test)
build_model.save_variables(path="./saved_variables/categories.pickle", variable=one_hot_encoder.categories_[0])
build_model.save_variables(path="./saved_variables/tf_idf_dict.pickle", variable=tf_idf_dict)
"""

"""
############################################
################ TESTING ###################
############################################
"""
#lOADING VARIABLES
X_test      =   build_model.load_variables(path="./saved_variables/X_test.pickle")
y_test      =   build_model.load_variables(path="./saved_variables/y_test.pickle")
categories  =   build_model.load_variables(path="./saved_variables/categories.pickle")
tf_idf_dict =   build_model.load_variables(path="./saved_variables/tf_idf_dict.pickle")

#LOADING THE MODEL
categories_classification_model = load_model("./ML_models/rental_incidents_classification_model.h5")

#PREDICTIONS ON TEST SET
X_test_cleansed, y_test_cleansed                =   build_model.get_meaningful_sentences_only_with_label(tf_idf_dict, X_test, y_test)
y_predicted, all_predictions                    =   build_model.predict(X_test_cleansed, categories_classification_model, categories)    
report                                          =   classification_report(y_true=y_test_cleansed, y_pred=y_predicted[:,0], output_dict=True)
report                                          =   pd.DataFrame(report).transpose()
report

#PREDICTION FOR A NEW OBSERVATION
sentence                    = "Bonjour je vous appelle car j'ai des punaises sur mes draps. Comment m'en débarrasser s'il vous plait?"
X_new                       = build_model.get_only_meaningful_sentences_without_label(tf_idf_dict, [sentence])
y_new_obs_predicted, allv   = build_model.predict(X_new, categories_classification_model, categories)    
y_new_obs_predicted




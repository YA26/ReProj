import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities  
from gensim.models import KeyedVectors
from nltk import RegexpTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd


class DataPreProcessing():

    def __init__(self, Word_2_Vec_path):
        self.__toknizer = RegexpTokenizer(r'''\w'|\w+|[^\w\s]''')
        self.__Word2_vec_model = KeyedVectors.load_word2vec_format(Word_2_Vec_path, binary=True)
        
    def get_rid_of_unwanted_chars(self, corpus):
        return " ".join(re.findall("[\D.]+", " ".join(re.findall("[\w.]+", corpus))))

    def scrape_google_search(self, query_and_labels, files_path, firefox_path, driver_path, pages):  
        '''
        - Function to scrape every link of n pages of a google search. 
        - In query_and_labels dataframe we have every query that we want to look for in google. We do also have their associated labels
        - To achieve the scraping, we use selenium (a firefox add-on that has been originally created for website testing) and BeautifulSoup
        '''
        cap                 = DesiredCapabilities.FIREFOX.copy() 
        options             = webdriver.FirefoxOptions()
        options.binary      = firefox_path
        cap["marionette"]   = True
        driver              = webdriver.Firefox(options=options,executable_path=r''+driver_path, capabilities=cap)
        all_contents        = []
        for value in query_and_labels.itertuples():     
            try:
                print("QUERY {} IS BEING TREATED...".format(value.query))
                driver.get("https://www.google.fr/search?q="+value.query+"&num="+str(pages*10))
                html = driver.page_source
                soup = BeautifulSoup(html)
                progression=0
                #1- We get every link on every selected page
                for div_tags in soup.find_all("div", class_="r"):
                    a_tags = div_tags.findChildren("a", recursive=False)[0].prettify()
                    link   = re.findall("https?://[\w./-]+",a_tags)[0]
                    driver.get(link)
                    html_link = driver.page_source
                    soup_link = BeautifulSoup(html_link)
                    #2- For every link, we get the content 
                    for div in soup_link.find_all("div", class_="content"):
                        text = self.get_rid_of_unwanted_chars(div.get_text())
                        if len(text.strip())>0:
                            all_contents.append([text, value.label])
                    progression+=1
                    print("{}-LINK: {} TREATED".format(progression, link))
                print("QUERY {} TREATED".format(value.query))
                progression=0         
            except (WebDriverException, UnicodeEncodeError):
                pass
            #3- We saved everything in a csv file
            data_scraped = pd.DataFrame(all_contents, columns=["content","label"])
            data_scraped.to_csv(path_or_buf=files_path, index=False)
    
    def train_test_separator(self, content, labels, test_size):
        '''
        Train_test splitter with stratify option enabled
        '''
        X_train, X_test, y_train, y_test = train_test_split(content, labels, test_size=test_size, stratify=labels)
        return X_train, X_test, y_train, y_test
    
    def sent_tokenizer(self, string):
        '''
        Self_made sentence tokenizer because sent_tokenizer of nltk doesn't split properly.
        Especially if there's no space after a full stop or other punctuations
        '''
        return re.split("[?.!]+", re.sub("[?.!]$", "", string))
    
    def group_contents_by_label(self, X, y):
        '''
        Function to group every single corpus in one corpora for every category
        '''
        X_total                = np.concatenate((X.reshape((-1,1)), y.reshape((-1,1))), axis=1)
        contents_by_categories = defaultdict(list)
        for row in X_total:
            content = row[0]
            label   = row[1]
            contents_by_categories[label].append(content)
        return contents_by_categories

    def clean_up_files(self, files_by_label):
        '''
        Function to get rid of everything that is not a word: unwanted punctuations, email-adresses and so forth...
        '''
        corpus_joined               = ""
        all_sentences               = ""
        unique_words                = None
        files_by_label_cleansed_up  = defaultdict(int)
        for label, corpora in files_by_label.items():
            if len(corpora)!=0:
                for corpus in corpora:
                    corpus_joined+="".join(corpus.lower())
                corpus_joined_tokenized             = self.sent_tokenizer(self.get_rid_of_unwanted_chars(corpus_joined))
                files_by_label_cleansed_up[label]   = corpus_joined_tokenized
                corpus_joined=""
                all_sentences+=" ".join(corpus_joined_tokenized)
        unique_words=list(sorted(set(self.__toknizer.tokenize(self.get_rid_of_unwanted_chars(all_sentences)))))
        return files_by_label_cleansed_up, unique_words

    def hist_corpus_per_label(self, files_by_label, path_save):
        '''
        Plot number of corpus per label
        '''
        #How many files per label
        x = list(files_by_label.keys())
        length_files_by_label=0 
        for corpus in list(files_by_label.keys()):
            length_files_by_label+=len(files_by_label[corpus])    
        y = [round((len(files_by_label[corpus])/length_files_by_label)*100,2) for corpus in list(files_by_label.keys())]
        plt.figure(figsize=(300,150)) #width:300  height:150
        plt.bar(x,y, align="center") # A bar chart
        plt.xlabel('Labels', fontsize=50, labelpad=40)
        plt.xticks(fontsize=150, rotation="vertical")
        plt.ylabel('Corpora', fontsize=50, labelpad=40)
        plt.yticks(fontsize=70)
        plt.title("Nombre de corpus par label", fontsize=100)
        for i in range(len(y)):
            plt.annotate(s=y[i],xy=(x[i], y[i]+1), fontsize=100)
            plt.hlines(y[i], xmin=0, xmax=0)         
        plt.savefig(path_save)
        plt.show()
  
    def tf_and_idf(self, files_by_label_cleansed_up):
        '''
        Term frequency and Inverse Document Frequency
        TF  = occurence of word w in a corpora
        IDF = log10(Total number of documents/ Number of documents with word w in it)   
        '''
        tf_dict                 = defaultdict(int) #term frequency of words per label
        N_doc_with_each_term    = defaultdict(int) #Number of documents(corpora) with word w in it 
        word_counts             = defaultdict(int) #unique_words_per_label
    
        for label, corpora in files_by_label_cleansed_up.items():
            #tf
            for sentence in corpora: 
                for word in self.__toknizer.tokenize(re.sub("[.]"," ",sentence)):
                    word_counts[word]+=1  
            tf_dict[label] = word_counts
            #idf
            for word in word_counts.keys():
                N_doc_with_each_term[word]+=1     
            word_counts = defaultdict(int)
        idf_dict={key: round(np.log10(len(files_by_label_cleansed_up.keys())/value), 2) for key, value in N_doc_with_each_term.items()}
        return tf_dict, idf_dict
    
    def cosine_similarity(self, list_of_words, cos_z_score_threshold, verbose=False):
        '''
        cosine similarity matrix of every word in a tf_idf[category] dictionary
        '''
        words=[]
        words_left_out=[]
        for word in list_of_words:
            #Checking if a vector for a given word exists. If it doesn't we discard the word
            try:   
                self.__Word2_vec_model.get_vector(word)
                words.append(word)
            except KeyError:
                words_left_out.append(word)     
        words_bis=words.copy()
        words_bis.append("TOTAL")
        #Initializing the cosine similarity matrix
        cosine_sim_matrix = pd.DataFrame({}, index=words, columns=words_bis)
        for counter, row in enumerate(cosine_sim_matrix.itertuples()):
            index   = row[0]
            cosines = []
            for word in cosine_sim_matrix.columns[:-1]:    
                index_vector    = self.__Word2_vec_model.get_vector(index)
                word_vector     = self.__Word2_vec_model.get_vector(word)
                cos             = np.dot(index_vector, word_vector)/(np.linalg.norm(index_vector)*np.linalg.norm(word_vector)) 
                #Cos is a function that has values ranging from [-1,1] we will only take into account values that are in [0,1]
                if cos<0:
                    cos=0
                cosine_sim_matrix.loc[index, word] = cos
                if index!=word:
                    cosines.append(cos)
            #We compute the mean of similarities. The greater it is the more similar the word is with other words.
            cosine_sim_matrix.loc[index, cosine_sim_matrix.columns[-1]] = np.mean(cosines)
            if verbose==True:
                print(str(counter)+" "+index+" treated")           
        #z_scores allow us to choose between discarding words that have no so similarity with others (for a given category) whatsoever(outliers) and those that are relatively a bit closer.   
        #positive z_score means that we only want to keep similar words
        z_scores=(cosine_sim_matrix.TOTAL - np.mean(cosine_sim_matrix.TOTAL))/np.std(cosine_sim_matrix.TOTAL)
        words_to_discard = list(z_scores[z_scores<cos_z_score_threshold].index)
        return words_to_discard, words_left_out

    def tf_idf_scores(self, tf_dict, idf_dict, tf_idf_z_score_threshold, cos_z_score_threshold, verbose=False): 
        '''
        tf_idf = tf * idf : it tells us how important a word is to a category
        But we need a second filter for our words: the cosine similarity
        We need words that are both important to a category and similar to other words for that same category 
        ''' 
        tf_idf_dict=defaultdict(int) 
        tf_idf_scores=defaultdict(int)
        for dict_label, documents in tf_dict.items():
            if verbose==True:
                print("\n***********{} IS BEING TREATED***********".format(dict_label))
            for word, term_frequency in documents.items():
                tf_idf_scores[word]=term_frequency*idf_dict[word]  
            #First filter:get the rarest words(the most influential ones) by keeping words that have a z_score greater than tf_z_score_threshold
            tf_idf_scores={word: (value - np.mean(list(tf_idf_scores.values()))) / np.std(list(tf_idf_scores.values())) for word, value in tf_idf_scores.items() if (value - np.mean(list(tf_idf_scores.values()))) / np.std(list(tf_idf_scores.values())) > tf_idf_z_score_threshold}
            
            #Second filter: get the words that are similar to one another 
            list_words=list(tf_idf_scores.keys())
            words_to_discard, words_left_out=self.cosine_similarity(list_words, cos_z_score_threshold, verbose)
            tf_idf_scores={word: value for word, value in tf_idf_scores.items() if word not in words_to_discard and word not in words_left_out}
            tf_idf_dict[dict_label]=tf_idf_scores
            tf_idf_scores=defaultdict(int)
        return tf_idf_dict
     
    def shrink_sentence(self, context_word, sentence, window):
        '''
        Function that shrink a sentence around a context word by a given a window size
        '''
        sent_index   = sentence.index(context_word)
        sentence_1   = " ".join(sentence[:sent_index].split()[-window:])
        sentence_2   = " ".join(sentence[sent_index+len(context_word):].split()[:window])
        new_sentence = "{} {} {}".format(sentence_1, context_word, sentence_2)
        return new_sentence

    def score_sentences(self, files_by_label_cleansed_up, tf_idf_dict, window_size):
        '''
        Scores every sentence for a given category.
        If the score exceeds the mean of all scores(for a given category), we keep the sentence in our final training set
        If the sentence is selected, we then need to shrink it around a specific context word (present in tf_idf or related to one of the words in a tf_idf)
        '''
        dict_scores=defaultdict(int)
        for label, sentences in files_by_label_cleansed_up.items():
            sentences_score_per_label=[]
            scores_per_label=[]
            for sentence in sentences:
                sentences_shrunk=[]
                n_words_per_sentence=0
                sentence_score=0
                for word in self.__toknizer.tokenize(sentence):
                    if word in tf_idf_dict[label].keys():
                        n_words_per_sentence+=1
                        word_score=tf_idf_dict[label][word]
                        sentence_score+=word_score
                        shrunk_sent=self.shrink_sentence(word, sentence, window_size)
                        if shrunk_sent not in sentences_shrunk:
                            sentences_shrunk.append(shrunk_sent)
                if n_words_per_sentence!=0:
                    sentence_score_normalized=sentence_score/n_words_per_sentence
                    scores_per_label.append(sentence_score_normalized)
                    for sentence_shrunk in sentences_shrunk:
                        sentences_score_per_label.append((sentence_shrunk, sentence_score_normalized))
            dict_scores[label]=[sentence for sentence in sentences_score_per_label if sentence[1] > np.mean(scores_per_label)]       
        return dict_scores
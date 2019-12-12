# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:45:45 2019

@author: BB
"""

import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
driver = webdriver.Firefox(executable_path=r'C:\Users\BB\Desktop\gecko_driver\geckodriver.exe')


query_and_labels = [("problème DAAF forum", "DAAF_electricite"),
                  ("problème équipement electrique appartement forum", "Equipements_specifiques_electrique_electricite"),
                  ("problème alimentation electrique appartement forum","alimentation_electrique_electricite"),
                  ("problème interrupteur forum","compteur_electrique_electricite"),
                  ("problème interphone forum","interphonie_electricite"),
                  ("problème compteur electrique forum","interrupteur_electricite"),
                  ("problème chauffage electrique forum","chauffage_electrique_electricite")]


def scrape_google_search(query_and_labels, path, pages):
    for label in query_and_labels:     
        try:
            print("QUERY {} IS BEING TREATED...".format(label[0]))
            driver.get("https://www.google.fr/search?q="+label[0]+"&num="+str(pages*10))
            html = driver.page_source
            soup = BeautifulSoup(html)
            links = []
            for div_tags in soup.find_all("div", class_="r"):
                a_tags = div_tags.findChildren("a", recursive=False)[0].prettify()
                links.append(re.findall("https?://[\w./-]+",a_tags)[0])
                
            content =[]
            progression=0
            for link in links:
                driver.get(link)
                html_link = driver.page_source
                soup_link = BeautifulSoup(html_link)
                for div in soup_link.find_all("div"):
                    content.append(div.get_text())
                progression+=1
                print("{}-LINK: {} TREATED".format(progression, link))
            print("QUERY {} TREATED".format(label[0]))
                
            with open(path+label[1]+".txt", 'w') as file:
                for item in content:
                    file.write("%s\n" % item)
            links = []
            content =[]
            progression=0
            print("FILE "+label[1]+".txt HAS BEEN SAVED TO: "+path+label[1]+".txt")
        except (WebDriverException, UnicodeEncodeError):
            pass
    


scrape_google_search(query_and_labels, "C:/Users/BB/Desktop/gecko_driver/files/", 2)


"content" in "ccm_question_full__question typo_content"

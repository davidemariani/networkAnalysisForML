
#simulator of corporation names
import random
import urllib.request
from numpy.random import choice, seed
import os
import requests


#upload Webster's Second International dictionary
word_url = "http://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"

BlackList = ["Negro", "negro", "Sex", "sex"]

#caching utils to avoid downloading from remote url on each call
def download_file(filename, url):
    with open(filename, 'wb') as fout:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        # Write response data to file
        for block in response.iter_content(4096):
            fout.write(block)
def download_if_not_exists(filename, url):
    if not os.path.exists(filename):
        print("DINE: file {:} not found in {:}, downloading...".format(filename, os.getcwd()))
        download_file(filename, url)
        return False
    return True

def toplural(word):
    if word[-1]=="y":
        plural = word[:-1]+"ies"
    elif word[-1] in ["s", "x", "z", "o", "h"]:
        plural = word+"es"
    elif word[-2:] == "us":
        plural = word[:-2]+"i"
    elif word[-2:] == "on":
        plural = word[:-2]+"a"
    else:
        plural = word+"s"
    return plural    

#run this to generate a list of N simulated names
def sim_names(N=1, pattc =  ([0, 0, 3], #two names, postfix
                             [0, 0, 2, 3], #two names, qualifier, postfix
                             [0, 3], #name, postfix
                             [0, 1, 3], #name, word, postfix
                             [0, 1, 4, 1, 3], #name, word and word, postfix
                             [5, 1, 3]), #adjective, word, postfix
              pattw =  [6, 2, 6, 4, 2, 2], 
              random_seed=42):
    """
    This function creates a random name or a set of N names
    patterns refer to elements of wordset list of lists below
    pattw are weights for each pattern
    """
    seed(random_seed)
    #get the dictionary unless it is already in the local dir
    download_if_not_exists("~Webster2.txt", word_url)
    with open("~Webster2.txt", "rt") as fle:
        words = fle.read().splitlines()

    #types of words
    namelikewords = [word for word in words if word[0].isupper() and not word in BlackList]
    nonnamewords = [word for word in words if word[0].islower() and not word in BlackList]
    pluralnnwords = [toplural(word) for word in nonnamewords]
    pluralnnwords = [word[0].upper() + word[1:] for word in pluralnnwords]

    wordset  = ([namelikewords, #uppercase starting, (0)
                pluralnnwords, #lowercase starting, plural (1)
                ["Brothers", "& Son", "United", "Industries", "Financial", "Holdings", "Global", 
                 "International", "Trading", "Supply", "Development", "Energy"], #can be before postfix (2)
                ['Ltd',  'S.p.a.', "SA", 'Limited', 'Sarl', 'GmbH', 'LLC', 'Inc.', #customary postfixes
                'Co.', 'Corporation'], #(3)
                ["and"], #(4)
                ["Global", "Universal", "United", "National", "First", "Second", 
                 "Authentic", "Federated", "Amalgamated", "Cooperative"] #(5)
               ])
    
    pattp = [x/sum(pattw) for x in pattw] #ensure probabilities sum to 1
    names=set()
    while len(names)<N:
        pattern = choice(pattc, p=pattp)
        names.add(" ".join(choice(wordset[i]) for i in pattern))
        
    return list(names)


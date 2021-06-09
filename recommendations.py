#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 13:07:21 2020

@author: asif
"""

#A dictionary of movie critics and their ratings of a small set of movies.


critics = {
        'Lisa Rose': {'Lady in the water':2.5, 'Snakes on a Plane':3.5,
                      'Just My Luck':3.0, 'Superman Returns':3.5,
                      'You, me and Dupree':2.5, 'The Night Listener':3.0},
        'Gene Seymour': {'Lady in the water':3.0, 'Snakes on a Plane':3.5,
                      'Just My Luck':1.5, 'Superman Returns':5.0,
                      'The Night Listener':3.0, 'You, me and Dupree':3.5},
        'Michael Phillips': {'Lady in the water':2.5, 'Snakes on a Plane':3.0,
                             'Superman Returns': 3.5, 'The Night Listener':4.0},
        'Claudia Puig': {'Snakes on a Plane':3.5, 'Just My Luck':3.0,
                         'The Night Listener':4.5, 'Superman Returns':4.0,
                         'You, me and Dupree':2.5},
        'Mick LaSalle': {'Lady in the water':3.0, 'Snakes on a Plane':4.0,
                         'Just My Luck':2.0, 'Superman Returns':3.0,
                         'The Night Listener':3.0, 'You, me and Dupree':2.0},
        'Jack Matthews': {'Lady in the water':3.0, 'Snakes on a Plane':4.0,
                          'The Night Listener':3.0, 'Superman Returns':5.0,
                          'You, me and Dupree':3.5},
        'Toby': {'Snakes on a Plane':4.5, 'You, me and Dupree':3.0,
                 'Superman Returns':4.0}
        }
        
from math import sqrt

#Returns a distance based similarity score for Person1 and Person2

def sim_distance(prefs,person1,person2):
    #Get the list of shared items
    si={}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item]=1
            
    #If they have no ratings in common, return 0
    if len(si)==0:
        return 0
    
    #Add up the squares of all the differences
    sum_of_squares=sum([pow(prefs[person1][item]-prefs[person2][item],2)
                        for item in prefs[person1] if item in prefs[person2]])
    return 1/(1+sum_of_squares)

sim_distance(critics, 'Lisa Rose', 'Gene Seymour')
sim_distance(critics, 'Toby', 'Mick LaSalle')

import matplotlib.pyplot as plt


#Returns the Pearson correlation coefficient for p1 and p2
def sim_peason(prefs,p1,p2):
    #Get the list of manually rated items
    si={}
    for item in prefs[p1]:
        if item in prefs[p2]: si[item]=1
    
    #Find the number of elements
    n=len(si)
    
    #If they are no ratings in common
    if n==0: return 0
    
    #Add up all the preferences
    sum1 = sum([prefs[p1][it] for it in si])
    sum2 = sum([prefs[p2][it] for it in si])
    
    #sum up the squares
    sum1Sq = sum([pow(prefs[p1][it],2) for it in si])
    sum2Sq = sum([pow(prefs[p2][it],2) for it in si])
    
    #Sum up the products
    pSum = sum([prefs[p1][it]*prefs[p2][it] for it in si])
    
    #calculate Pearson Score
    num = pSum-(sum1*sum2/n)
    den = sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
    if den==0: return 0
    
    r = num/den
    
    return r

sim_peason(critics, 'Lisa Rose', 'Gene Seymour')


#RETURNS THE BEST MATCHES FOR PERSON FROM THE PREFS DICTIONARY
#NUMBER OF RESULTS AND SIMILARITY FUNCTION ARE OPTIONAL PARAMS.
def topMatches(prefs, person, n=5, similarity=sim_peason):
    scores = [(similarity(prefs, person, other), other) for other in prefs if other!=person]
    
    #Sort of list so the highest scores appear at the top
    scores.sort()
    scores.reverse()
    return scores[0:n]

topMatches(critics, 'Toby', n=3)


#GETS THE RECOMMENDATIONS FOR A PERSON BY USING A WEIGHTED AVERAGE
#OF EVERY OTHER USER'S RANKING
def getRecommendations(prefs, person, similarity=sim_peason):
    totals={}
    simSums={}
    for other in prefs:
        #don't compare me to myself
        if other==person: continue
        sim=similarity(prefs, person, other)
        
        #ignore scores of zero or lower
        if sim<=0: continue
        for item in prefs[other]:
            
            #only score movies I haven't seen yet
            if item not in prefs[person] or prefs[person][item]==0:
                #similarity * 0
                totals.setdefault(item, 0)
                totals[item]+=prefs[other][item]*sim
                #sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+=sim
                
        #Create the normalized list
        rankings = [(total/simSums[item],item) for item, total in totals.items()]
        
        #Return the sorted list
        rankings.sort()
        rankings.reverse()
        return rankings


getRecommendations(critics, 'Toby')


def transformPrefs(prefs):
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})
            
            #Flip item and person
            
        result[item][person]=prefs[person][item]
    return result

movies = transformPrefs(critics)
topMatches(movies, 'Superman Returns')


bDict = {val:key for (key, val) in critics.items()}
print(bDict)    # {'Seven': 7, 'Two': 2, 'Ten': 10}



def calculateSimilarItems(prefs, n=10):
    #Create a dictionary of items showing which other items they
    #are most similar to
    result={}
    
    #Inverse the preference matrix to be item-centric
    itemPrefs = transformPrefs(prefs)
    c=0
    for item in itemPrefs:
        #Status updates for large datasets
        c+=1
        if c%100==0: print("%d / %d" % (c, len(itemPrefs)))
        #Find the most similar to this one
        scores = topMatches(itemPrefs, item, n=n, similarity=sim_distance)
        result[item]=scores
    return result

itemsim=calculateSimilarItems(critics)


                
                
#WORKING WITH MOVIELENS
def loadMovieLens(path='/Users/asif/Downloads/ml-100k'):
    
    #Get movie titles
    movies={}
    for line in open(path+'/u.item', encoding = "ISO-8859-1"):
        (id, title)=line.split('|')[0:2]
        movies[id]=title
        
    #Load Data
    prefs={}
    for line in open(path+'/u.data', encoding = "ISO-8859-1"):
        (user,movieid,rating,ts)=line.split('\t')
        prefs.setdefault(user,{})
        prefs[user][movies[movieid]]=float(rating)
    return prefs

prefs = loadMovieLens()
prefs['87']

getRecommendations(prefs, '87')[0:30]

itemsim = calculateSimilarItems(prefs,n=50)

getRecommendations(prefs, itemsim, '87')[0:30]
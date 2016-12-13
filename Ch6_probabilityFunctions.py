# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 07:19:16 2016

@author: ross

Data Science From Scratch
Chapter 6: Probability
Independence
"""
from __future__ import division
from collections import Counter
import math, random


def random_kid():
    return random.choice(["boy", "girl"])
    
both_girls = 0
older_girl = 0
either_girl = 0
    
random.seed(0)
for _ in range(10000):
    younger = random_kid()
    older = random_kid()
    if older == "girl":
        older_girl += 1
    if older == "girl" and younger == "girl":
        both_girls += 1
    if older == "girl" or younger == "girl":
        either_girl += 1
    
print "P(both | older): ", both_girls / older_girl

print "P(both | either): ", both_girls / either_girl


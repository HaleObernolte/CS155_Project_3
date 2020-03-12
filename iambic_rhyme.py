#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 04:53:25 2020

@author: Yacong
"""


import nltk
import numpy as np
import string
import sonnet_processing as sp
import HMM

############## Use CMU’s Pronouncing Dictionary in NLTK ######################

def get_stress(word):
    syllables =nltk.corpus.cmudict.dict()[word]

    if syllables:
        # just picking the first stress ?????
        pronunciation_string = ''.join(syllables[0])
        # Not interested in secondary stress
        stress_numbers = ''.join([x.replace('2', '1')
                                  for x in pronunciation_string if x.isdigit()])
    return stress_numbers


# put this in sonnet_processing.py ???
def get_stress_map(fname):
    with open(fname,'r') as f:
        stress_map = {}
        for line in f:
            fields = line.split()
            word = fields[0]
            word = word.translate(str.maketrans('', '', string.punctuation))
            word = word.lower()
            stresses = get_stress(word)
            if word not in stress_map:
                stress_map[word] = stresses
    return stress_map

##############################################################################
    
##### or learn the stress map from data (sonnets are iambic pentameter #######
# still not right. need to count stress for each syllable, not each word!!!
def get_stress_map(fname,max_words=4000):
    with open(fname,'r') as f:
        # Skip first sonnet number
        f.readline()
        # Parse sonnets
        obs_counter = 0
        stressed_count = {}
        unstressed_count = {}
        
        stress_map = {}
        while (obs_counter < max_words):
            cont = True
            while (True):
                # get line
                line = f.readline()
                # Check for eof
                if not line:
                    cont = False
                    break
                # Check for end of sonnet
                if (len(line) == 1):
                    # Skip next length 1 lines
                    f.readline()
                    f.readline()
                    break
                words = line.split()
                for i_word,raw_word in enumerate(words):
                    word = raw_word.translate(str.maketrans('', '', string.punctuation))
                    word = word.lower()
                    if i_word%2 == 1:
                        if word not in stressed_count:
                            # Add unique words to the stress map.
                            stressed_count[word] = 1
                        else:
                            stressed_count[word] += 1
                    else:
                        if word not in unstressed_count:
                            # Add unique words to the stress map.
                            unstressed_count[word] = 1
                        else:
                            unstressed_count[word] += 1
            # Check for EOF
            if (not cont):
                break   
        for word in stressed_count:
            stress_map[word] = round(stressed_count[word] / (stressed_count[word] \
                                                        + unstressed_count[word]))
        return stress_map


##############################################################################
    
    

def check_iambic(words, stress_map):
    is_iambic = True
    current_stress = 0
    for i_word,word in enumerate(words):
        word_stresses = stress_map[word]
        if i_word == 0:
            if word_stresses[0] != 0:
                is_iambic = False
                break
            current_stress = word_stresses[-1]
        else:
            next_stress =  word_stresses[0]
            if next_stress == current_stress:
                is_iambic = False
                break
            else:
                current_stress = word_stresses[-1]

    return is_iambic


def add_word(current_state,A,O):
    num_states = len(A)
    num_tokens = len(O[0])
    # Select random token from current state based on emission matrix.
    token = int(np.random.choice(num_tokens, p=O[int(current_state)]))
    current_state = np.random.choice(num_states, p=A[int(current_state)])
    return token,current_state

def add_word_backward(current_state,A,O):
    num_states = len(A)
    num_tokens = len(O[0])
    # Select random token from current state based on emission matrix.
    token = int(np.random.choice(num_tokens, p=O[int(current_state)]))
    # Backwards generate previous state
    t_probs = []
    for s in range(num_states):
        t_probs.append(A[s][current_state])
    t_sum = sum(t_probs)
    for i in range(num_states):
        t_probs[i] = t_probs[i] / t_sum
    prev_state = np.random.choice(num_states, p=t_probs)
    return token,prev_state    


def syllable_count(words,syllable_map):
    count = 0
    for word in words:
        count += syllable_map[word]
    return count


def make_rhyme_iambic_line(rhyme_word,syllable_map,stress_map,length=10):
    words = [rhyme_word]
    current_state = get_state(rhyme_word)
    while (syllable_count(words, syllable_map) != 10 or
            check_iambic(words, stress_map, syllable_map) is not True):
        while syllable_count(words, syllable_map) < 10:
            token, current_state = add_word_backward(current_state)
            words += token
    words.reverse()
    line = ""
    for word in words:
        line = line + word + ' '
    return line


def make_iambic_quatrain(two_rhymes,stress_map):
    quatrain = ""
    quatrain += make_rhyme_iambic_line(two_rhymes[0][0],length=10) + '\n'
    quatrain += make_rhyme_iambic_line(two_rhymes[1][0],length=10) + '\n'
    quatrain += make_rhyme_iambic_line(two_rhymes[0][1],length=10) + '\n'
    quatrain += make_rhyme_iambic_line(two_rhymes[1][1],length=10) + '\n'
    return quatrain


def make_iambic_couplet(one_rhyme,stress_map):
    couplet = ""
    couplet += make_rhyme_iambic_line(one_rhyme[0],length=10) + '\n'
    couplet += make_rhyme_iambic_line(one_rhyme[1],length=10) + '\n'
    return couplet


def make_rhyme_iambic_sonnet(rhymes,stress_map):
    
    # abab cdcd efef gg
    rhymes_abcdefg = []
    while(len(rhymes_abcdefg) < 6):
        new_rhyme = rhymes[np.random.randint(len(rhymes))]
        # avoid duplicated rhyme pairs
        if new_rhyme not in rhymes_abcdefg:
            rhymes_abcdefg.append(new_rhyme)
            
    sonnet = ""
    for i_quatrain in range(3):
        sonnet += make_iambic_quatrain(rhymes_abcdefg[i_quatrain*2:(i_quatrain+1)*2],
                                       stress_map)
        sonnet += '\n'
    sonnet += make_iambic_couplet(rhymes_abcdefg[6],stress_map)
    
    return sonnet

    

data_path = 'data/'
rhymes = sp.get_rhymes(data_path+'shakespeare.txt')

# choose 1 from below
stress_map = get_stress_map(data_path+'shakespeare.txt')
stress_map = get_stress_map(data_path+'Syllable_dictionary.txt')

sonnet = make_rhyme_iambic_sonnet(rhymes,stress_map)
print(sonnet)
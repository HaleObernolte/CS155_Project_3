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


##############################################################################
##############             Get stress map               ######################
############## Use CMU’s Pronouncing Dictionary in NLTK ######################
##############################################################################

def get_stress(word):
    """
    Get stress of a single word (secondary stress is also treated as stress).

    Parameters
    ----------
    word : str
        The word to be analyzed.

    Returns
    -------
    all_possible_stress_numbers : list
        List of all possible stresses (as strings os 0s and 1s).

    """
    
    try:
        syllables = nltk.corpus.cmudict.dict()[word]
    except KeyError:
        # can make this more elegant by guessing the original word from given
        return ''

    if syllables:
        all_possible_stress_numbers = []
        # analysis all possible pronunciations of the word
        for pronunciation in syllables:
            pronunciation_string = ''.join(pronunciation)
            # secondary stress is also a stress
            # 1=primary, 2=secondary, 0=no stress
            stress_numbers = ''.join([x.replace('2', '1')
                                      for x in pronunciation_string if x.isdigit()])
            if stress_numbers not in all_possible_stress_numbers:
                all_possible_stress_numbers.append(stress_numbers)
        
    return all_possible_stress_numbers


# can put this in sonnet_processing.py 
def get_stress_map_nltk(fname):
    """
    Generate stress map from file using NLTK.

    Parameters
    ----------
    fname : str
        File containing all words in the sonnet data (Syllable_dictionary.txt).

    Returns
    -------
    stress_map : dict
        key = word. value = list of stress sequences.

    """
    with open(fname,'r') as f:
        stress_map = {}
        for line in f:
            fields = line.split()
            # don't consider lines of sonnet indices
            if len(fields) > 1:
                word = fields[0]
                word = word.translate(str.maketrans('', '', string.punctuation))
                word = word.lower()
                stresses = get_stress(word)
                if word not in stress_map:
                    stress_map[word] = stresses
    return stress_map



##############################################################################
##############             Get stress map               ######################
######  learn the stress map from data (sonnets are iambic pentameter)  ######
##############################################################################
    
def get_stress_map(fname, max_words=7000):
    """
    Generate stress map from file.

    Parameters
    ----------
    fname : str
        File containing all sonnets (shakespeare.txt).
    max_words : int, optional
        Max number of words in the stress map. The default is 7000.
        Each pronunciation is one entry. Each word can have more than one entries.

    Returns
    -------
    stress_map : dict
        key = word+n_stress(int). value = np array of stress frequencies.

    """
    
    full_syl_map = get_full_syllable_map('../data/Syllable_dictionary.txt')
    
    with open(fname,'r') as f:
        # Skip first sonnet number
        f.readline()
        # Parse sonnets
        obs_counter = 0
        
        stressed_array = '1010101010'
        unstressed_array = '0101010101'
        
        stress_map = {}
        stress_map_counter = {}
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
                if len(words) > 1:
                    # get number of stresses per word in this line
                    n_stesses = match_stress(words,full_syl_map,n_total_stresses=10)
                    # iambic pentameter always start with unstressed
                    current_stress = '0'
                    for i_word,raw_word in enumerate(words):
                        # process word
                        word = raw_word.translate(str.maketrans('', '', string.punctuation))
                        word = word.lower()
                        # determine the stresses of this word
                        n_stress = n_stesses[i_word]
                        if current_stress == '0':
                            stresses = stressed_array[n_stress]
                        elif current_stress == '1':
                            stresses = unstressed_array[n_stress]
                        current_stress = stresses[-1]
                        # convert format of stresses from str to list of int
                        stress_int_list = []
                        for i in stresses:
                            stress_int_list.append(int(i))
                        # store this appearance of stress of this word to stress map
                        # word has an int (n_stress) appended at the end
                        if word+str(n_stress) not in stress_map:
                            stress_map[word+str(n_stress)] = np.array(stress_int_list)
                            stress_map_counter[word+str(n_stress)] = 1
                            obs_counter += 1
                        else:
                            stress_map[word+str(n_stress)] += np.array(stress_int_list)
                            stress_map_counter[word+str(n_stress)] += 1
                        
            # Check for EOF
            if (not cont):
                break   
    for word in stress_map.keys():
        stress_map[word] = stress_map[word] / stress_map_counter[word]
    return stress_map


def get_full_syllable_map(f_name):
    """
    Generate a syllable map where all possible pronunciations are included.

    Parameters
    ----------
    f_name : str
        File containing all words in the sonnet data (Syllable_dictionary.txt).

    Returns
    -------
    full_syl_map : dict
        key = words. values = list of int of stresses (e.g., ['E2', '3'])

    """
    f = open(f_name, "r")
    full_syl_map = {}
    for line in f:
        words = line.split()
        w = words[0]
        w = w.translate(str.maketrans('', '', string.punctuation))
        w = w.lower()
        stresses = words[1:]
        if w not in full_syl_map:
            full_syl_map[w] = stresses
    return full_syl_map


def match_stress(words,full_syl_map,n_total_stresses=10):
    """
    Find the combination of stresses for words that sum up to n_total_stresses.
    'word' is a sentence and end is considered.

    Parameters
    ----------
    words : list
        list of words (str).
    full_syl_map : dict
        See in get_full_syllable_map.
    n_total_stresses : int, optional
        The total number of stresses. The default is 10.

    Returns
    -------
    n_stresses : list
        list of number(int) of stresses corresponding to each word in words.

    """
    n_stresses = []
    words.reverse()
    last_syllable_specific = False
    
    # start from last word
    raw_word = words[0]
    word = raw_word.translate(str.maketrans('', '', string.punctuation))
    word = word.lower()
    syllables = full_syl_map[word]
    for syllable in syllables:
        if 'E' in syllable: # n_syl of last word is specific
            last_stress = int(syllable[1:])
            n_stresses.append(last_stress)
            n_stresses += match_stress_recursion(words[1:],
                                                 full_syl_map,
                                                 n_total_stresses-last_stress)
            last_syllable_specific = True
            break
    if not last_syllable_specific: # n_syl of last word has choices
        n_stresses = match_stress_recursion(words,
                                            full_syl_map,
                                            n_total_stresses)
    return n_stresses
    

def match_stress_recursion(words,full_syl_map,n_total_stresses):
    """
    Find the combination of stresses for words that sum up to n_total_stresses.
    'word' is simply a list and sentence end is NOT considered.

    Parameters
    ----------
    words : list
        list of words (str)..
    full_syl_map : dict
        See in get_full_syllable_map.
    n_total_stresses : list
        The total number of stresses in the given list of words.

    Returns
    -------
    n_stresses : list
        See match_stress.

    """
    n_stresses = []
    
    if len(words) > 1:
        raw_word = words[0]
        word = raw_word.translate(str.maketrans('', '', string.punctuation))
        word = word.lower()
        syllables = full_syl_map[word]
        for syllable in syllables:
            if 'E' not in syllable: 
                last_stress = int(syllable)
                n_stresses.append(last_stress)
                n_stresses += match_stress_recursion(words[1:],
                                                     full_syl_map,
                                                     n_total_stresses-last_stress)
    elif len(words) == 1:
        raw_word = words[0]
        word = raw_word.translate(str.maketrans('', '', string.punctuation))
        word = word.lower()
        syllables = full_syl_map[word]
        for syllable in syllables:
            if 'E' not in syllable: 
                last_stress = int(syllable)
                n_stresses.append(last_stress)
        
    return n_stresses


##############################################################################
##############        Build sonnets from words          ######################
##############################################################################

def check_iambic_pentameter(words, obs_map_r, stress_map, syllable_map):
    """
    Evaluate the quality of generated line by calculate the Euclidean distance 
    between standard iambic pentameter and generated line.

    Parameters
    ----------
    words : list
        List of words (str).
    obs_map_r : dict
        key = word ID (int). value = word (str).
    stress_map : dict
        key = word (str). value = word ID (int).
    syllable_map : dict
        key = word (str). value = number of stresses (int).

    Returns
    -------
    float
        If the line has 10 stresses: the Euclidean distance (unnormalized).
        If the line does NOT have 10 stresses: 2.

    """
    # not necessarily 0101010101. only need transition b/w words are 1-0 or 0-1
    # single-syllable words can be either 0 or 1
    # report the Euclidean distance b/w stresses of generated line and 0101010101
    standard_seq = np.array([0,1,0,1,0,1,0,1,0,1])
    stress_seq = np.array([])
    for word in words:
        n_stress = syllable_map[obs_map_r[word]]
        stress_seq = np.append(stress_seq, stress_map[obs_map_r[word]+str(n_stress)])
    if len(stress_seq) != 10:
        # has more than 10 syllables
        return 2
    else:
        return sum(np.square(stress_seq - standard_seq))


def add_word(current_state,A,O):
    """
    Add a word to a line of poem.

    Parameters
    ----------
    current_state : int
        Current state.
    A : numpy array
        Transition matrix.
    O : numpy array
        Observation matrix.

    Returns
    -------
    token : int
        ID of a new word.
    current_state : int
        Current state.

    """
    num_states = len(A)
    num_tokens = len(O[0])
    # Select random token from current state based on emission matrix.
    token = int(np.random.choice(num_tokens, p=O[int(current_state)]))
    current_state = np.random.choice(num_states, p=A[int(current_state)])
    return token,current_state

def add_word_backward(current_state,A,O):
    """
    Add a word to poem from backwards.

    Parameters
    ----------
    current_state : int
        Current state.
    A : numpy array
        Transition matrix.
    O : numpy array
        Observation matrix.

    Returns
    -------
    token : int
        ID of a new word.
    prev_state : float
        Previous state.

    """
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


def syllable_count(words,obs_map_r,syllable_map):
    """
    Count total number of syllables in sentence 'words'.

    Parameters
    ----------
    words : list
        List strings.
    obs_map_r : dict
        See check_iambic_pentameter.
    syllable_map : TYPE
        See check_iambic_pentameter.

    Returns
    -------
    count : int
        Number of syllables.

    """
    count = 0
    for word in words:
        count += syllable_map[obs_map_r[word]]
    return count


def get_state(word, obs_map, O):
    """
    Get most porbable state of current word.

    Parameters
    ----------
    word : str
        The word we need..
    obs_map : dict
        key = word. value = word ID.
    O : numpy array
        Observation matrix.

    Returns
    -------
    max_state : int
        The most probable state.

    """
    max_prob = 0
    max_state = -1
    for s in range(len(O)):
        p = O[s][obs_map[word]]
        if (p > max_prob):
            max_prob = p
            max_state = s
    return max_state


def make_rhyme_iambic_line(rhyme_word,obs_map,obs_map_r,syllable_map,stress_map,A,O,length=10):
    """
    Generate a new line that satisfies rhyme and iambic pentameter.

    Parameters
    ----------
    rhyme_word : str
        Last word of the line, for rhyme.
    obs_map : dict
        See get_state.
    obs_map_r : dict
        See check_iambic_pentameter.
    syllable_map : dict
        See check_iambic_pentameter.
    stress_map : dict
        See check_iambic_pentameter.
    A : numpy array
        Transition matrix.
    O : numpy array
        Emission matrix.
    length : int, optional
        Total number of syllables in the new line. The default is 10.

    Returns
    -------
    line : str
        The new sonnet line.

    """
    words = [obs_map[rhyme_word]]
    current_state = get_state(rhyme_word,obs_map,O)
    while (check_iambic_pentameter(words, obs_map_r, stress_map, syllable_map) > 1):
        while syllable_count(words, obs_map_r, syllable_map) < 10:
            token, current_state = add_word_backward(current_state,A,O)
            words += [token]
    words.reverse()
    line = ""
    for word in words:
        line = line + word + ' '
    return line


def make_iambic_quatrain(two_rhymes,obs_map,obs_map_r,stress_map,syllable_map,A,O):
    """
    Generate a new quatrain that satisfies rhyme and iambic pentameter.

    Parameters
    ----------
    two_rhymes : list
        list of list of rhyme word pairs (2 pairs, 4 words).
    obs_map : dict
        See get_state.
    obs_map_r : dict
        See check_iambic_pentameter.
    stress_map : dict
        See check_iambic_pentameter.
    syllable_map : dict
        See check_iambic_pentameter.
    A : numpy array
        Transition matrix.
    O : numpy array
        Emission matrix.

    Returns
    -------
    quatrain : str
        The new quatrain.

    """
    quatrain = ""
    quatrain += make_rhyme_iambic_line(two_rhymes[0][0],
                                       obs_map,obs_map_r,syllable_map,stress_map,
                                       A,O,length=10) + '\n'
    quatrain += make_rhyme_iambic_line(two_rhymes[1][0],
                                       obs_map,obs_map_r,syllable_map,stress_map,
                                       A,O,length=10) + '\n'
    quatrain += make_rhyme_iambic_line(two_rhymes[0][1],
                                       obs_map,obs_map_r,syllable_map,stress_map,
                                       A,O,length=10) + '\n'
    quatrain += make_rhyme_iambic_line(two_rhymes[1][1],
                                       obs_map,obs_map_r,syllable_map,stress_map,
                                       A,O,length=10) + '\n'
    return quatrain


def make_iambic_couplet(one_rhyme,obs_map,obs_map_r,stress_map,syllable_map,A,O):
    """
    Generate a new couplet that satisfies rhyme and iambic pentameter.

    Parameters
    ----------
    one_rhyme : list
        List of a rhyme word pair (1 pairs, 2 words).
    obs_map : dict
        See get_state.
    obs_map_r : dict
        See check_iambic_pentameter.
    stress_map : dict
        See check_iambic_pentameter.
    syllable_map : dict
        See check_iambic_pentameter.
    A : numpy array
        Transition matrix.
    O : numpy array
        Emission matrix.

    Returns
    -------
    couplet : str
        The new couplet.

    """
    couplet = ""
    couplet += make_rhyme_iambic_line(one_rhyme[0],
                                      obs_map,obs_map_r,syllable_map,stress_map,
                                      A,O,length=10) + '\n'
    couplet += make_rhyme_iambic_line(one_rhyme[1],
                                      obs_map,obs_map_r,syllable_map,stress_map,
                                      A,O,length=10) + '\n'
    return couplet


banned_end_words = ['the', 'a', 'an', 'at', 'been', 'in', 'of', 'to', 'by', 
                    'my', 'too', 'not', 'and', 'but', 'or', 'than', 'then', 
                    'no', 'o', 'for', 'so', 'which', 'their',  'on', 'your', 
                    'as', 'has', 'what', 'is', 'nor', 'i']

def make_rhyme_iambic_sonnet(rhymes,obs_map,obs_map_r,stress_map,syllable_map,A,O):
    """
    Generate a Shakespearen style sonnet that satisfies rhyme and iambic pentameter.

    Parameters
    ----------
    rhymes : list
        List of tuples of rhyme word pairs.
    obs_map : dict
        See get_state.
    obs_map_r : dict
        See check_iambic_pentameter.
    stress_map : dict
        See check_iambic_pentameter.
    syllable_map : dict
        See check_iambic_pentameter.
    A : numpy array
        Transition matrix.
    O : numpy array
        Emission matrix.

    Returns
    -------
    sonnet : str
        The new sonnet.

    """
    
    # abab cdcd efef gg
    rhymes_abcdefg = []
    while(len(rhymes_abcdefg) < 6):
        new_rhyme = rhymes[np.random.randint(len(rhymes))]
        if new_rhyme[0] not in banned_end_words and new_rhyme[0] not in banned_end_words:
            # avoid duplicated rhyme pairs
            if new_rhyme not in rhymes_abcdefg:
                rhymes_abcdefg.append(new_rhyme)
            
    sonnet = ""
    for i_quatrain in range(3):
        sonnet += make_iambic_quatrain(rhymes_abcdefg[i_quatrain*2:(i_quatrain+1)*2],
                                       obs_map,obs_map_r,
                                       stress_map,syllable_map,A,O)
        sonnet += '\n'
    sonnet += make_iambic_couplet(rhymes_abcdefg[6],obs_map,obs_map_r,
                                  stress_map,syllable_map,A,O)
    
    return sonnet


##############################################################################
#################    Model training and Sonnet Generation    #################
##############################################################################

# data_path = 'data/'
data_path = '../data/'

rhymes = sp.get_rhymes(data_path+'shakespeare.txt')

# choose 1 from below. suggest get_stress_map
stress_map = get_stress_map(data_path+'shakespeare.txt')
# stress_map = get_stress_map_nltk(data_path+'Syllable_dictionary.txt')

# choose 1 from below. suggest sp.get_syllable_map
syllable_map = sp.get_syllable_map(data_path+"Syllable_dictionary.txt") 
# syllable_map = get_full_syllable_map(data_path+'Syllable_dictionary.txt')

sonnets, obs_map = sp.get_sonnets(data_path+"shakespeare.txt", 1000)
model = HMM.unsupervised_HMM(sonnets, 10, 10)
obs_map_r = {}
for key in obs_map:
    obs_map_r[obs_map[key]] = key

sonnet = make_rhyme_iambic_sonnet(rhymes,obs_map,obs_map_r,stress_map,syllable_map,
                                  model.A,model.O)
print(sonnet)
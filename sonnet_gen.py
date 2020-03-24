#########################################
# sonnet_gen.py                         #
#                                       #
# Generates sonnets using given models  #
#########################################

############
# Imports  #
############

import sonnet_processing as sp
import HMM
import random
import numpy as np
import pandas as pd
import string


#####################
# Helper functions  #
#####################

# Returns the number of syllables in a line
def count_syllables(line, syl_map):
    count = 0
    words = line.split()
    for word in words:
        if word not in syl_map:
            print("ERROR:", word, "not found in syllable map.")
        count += syl_map[word]
    return count

# Returns a 10 syllable substring of a line, if it exists
def get_n_syls(n, line, syl_map):
    out = ""
    syls = 0
    words = line.split()
    for word in words:
        if (syls == n):
            return out
        if (syls > n):
            return ""
        out += word
        out += " "
        if word not in syl_map:
            print("ERROR:", word, "not found in syllable map.")        
        syls += syl_map[word]
    return ""

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

def get_state(word, O):
    max_prob = 0
    max_state = -1
    for s in range(len(O)):
        p = O[s][word]
        if (p > max_prob):
            max_prob = p
            max_state = s
    return max_state

def make_rhyme_line(model, r_w, syl_map, obs_map_r):
    line = ""
    rhyme_word = obs_map_r[r_w]
    while (not line):
        l = [rhyme_word]
        curr_state = get_state(r_w, model.O)
        no_syls = syl_map[rhyme_word]
        while (no_syls < 10):
            prev_word, curr_state = add_word_backward(curr_state, model.A,  model.O)
            w = obs_map_r[prev_word]
            l.append(w)
            no_syls += syl_map[w]
        if (no_syls == 10):
            l.reverse()
            for word in l:
                line += word
                line += " "
    return line.capitalize()
        
    


###########################
# Shakespeare generation  #
###########################

# Generates one sheakespearian sonnet using a HMM
def hmm_shakespeare_sonnet_naive():
    # Load in everything
    sonnets, obs_map = sp.get_sonnets("data/shakespeare.txt", 900)
    obs_map_r = {}
    for key in obs_map:
        obs_map_r[obs_map[key]] = key
    # Train HMM
    model = HMM.unsupervised_HMM(sonnets, 10, 10)
    # Print one blank line to make it pretty
    print("")
    # Generate quatrain 1
    for _ in range(4):
        line = ""
        while (not line):
            l = ""
            emission, states = model.generate_emission(7)
            for i in range(len(emission)):
                e = emission[i]
                w = obs_map_r[e]
                if (i == 0):
                    w = w.capitalize()
                l += w
                l += " "
                line = l
        print(line)
    print("")
    # Generate quatrain 2
    for _ in range(4):
        line = ""
        while (not line):
            l = ""
            emission, states = model.generate_emission(7)
            for i in range(len(emission)):
                e = emission[i]
                w = obs_map_r[e]
                if (i == 0):
                    w = w.capitalize()
                l += w
                l += " "
            line = l
        print(line)
    # Generate couplet
    for _ in range(2):
        line = ""
        while (not line):
            l = ""
            emission, states = model.generate_emission(7)
            for i in range(len(emission)):
                e = emission[i]
                w = obs_map_r[e]
                if (i == 0):
                    w = w.capitalize()
                l += w
                l += " "
            line = get_n_syls(10, l, syl_map)
        print(line)
        
# Generates one sheakespearian sonnet using a HMM
#    GOALS: Retains state from line to line to hopefully be more meaningful
#           Also restricts each line to be 10 syllables
def hmm_shakespeare_sonnet_goal1():
    # Load in everything
    sonnets, obs_map = sp.get_sonnets("data/shakespeare.txt", 3000)
    obs_map_r = {}
    for key in obs_map:
        obs_map_r[obs_map[key]] = key
    syl_map = sp.get_syllable_map("data/Syllable_dictionary.txt")
    # Train HMM
    model = HMM.unsupervised_HMM(sonnets, 5, 25)
    num_states = model.L
    # Print one blank line to make it pretty
    print("")
    # Generate sonnet
    last_state = np.random.choice(num_states, p=model.A_start)
    for n_lines in [4, 4, 2]:
        for l_no in range(n_lines):
            line = ""
            while (not line):
                curr_state = last_state
                l = ""
                no_syls = 0
                while (no_syls < 10):
                    w, curr_state = add_word(curr_state, model.A, model.O)
                    new_word = obs_map_r[w]
                    l += new_word
                    l += " "
                    no_syls += syl_map[new_word]
                if (count_syllables(l, syl_map) == 10):
                    line = l
            last_state = curr_state
            print(line.capitalize())
        print("") 
        
        
# Generates one sheakespearian sonnet using a HMM
#    GOALS: Seeds end of line with rhyme and generates backwards
#           Also restricts each line to be 10 syllables
def hmm_shakespeare_sonnet_goal2():
    # Load in everything
    sonnets, obs_map = sp.get_sonnets("data/shakespeare.txt", 2000)
    obs_map_r = {}
    for key in obs_map:
        obs_map_r[obs_map[key]] = key
    syl_map = sp.get_syllable_map("data/Syllable_dictionary.txt")
    rhymes = sp.get_rhymes("data/shakespeare.txt", True)
    # Train HMM
    model = HMM.unsupervised_HMM(sonnets, 5, 25)
    num_states = model.L
    # Print one blank line to make it pretty
    print("")
    # Generate quatrains
    for _ in range(2):
        while (True):
            r = random.randint(0, len(rhymes) - 1)
            rhyme_pair_1 = rhymes[r]
            if (rhyme_pair_1[0] in obs_map and rhyme_pair_1[1] in obs_map):
                break
        while (True):
            r = random.randint(0, len(rhymes) - 1)
            rhyme_pair_2 = rhymes[r]
            if (rhyme_pair_2[0] in obs_map and rhyme_pair_2[1] in obs_map):
                break
        print(make_rhyme_line(model, obs_map[rhyme_pair_1[0]], syl_map, obs_map_r))
        print(make_rhyme_line(model, obs_map[rhyme_pair_2[0]], syl_map, obs_map_r))
        print(make_rhyme_line(model, obs_map[rhyme_pair_1[1]], syl_map, obs_map_r))
        print(make_rhyme_line(model, obs_map[rhyme_pair_2[1]], syl_map, obs_map_r))
        print("")
    # Generate couplet
    while (True):
        r = random.randint(0, len(rhymes) - 1)
        rhyme_pair = rhymes[r]
        if (rhyme_pair[0] in obs_map and rhyme_pair[1] in obs_map):
            break
    print(make_rhyme_line(model, obs_map[rhyme_pair[0]], syl_map, obs_map_r))
    print(make_rhyme_line(model, obs_map[rhyme_pair[1]], syl_map, obs_map_r))
    print("")    
        
        
# Generates one sheakespearian sonnet using a HMM
#    GOALS: Retains state from line to line to hopefully be more meaningful
#           Also uses Star Wars Episode IV script
def hmm_shakespeare_sonnet_goal3():
    # Load in everything
    sonnets, obs_map = sp.get_sonnets("data/shakespeare.txt", 1000)
    obs_counter = len(obs_map)
    star_wars_lines = []
    f = open("data/StarWarsIV.txt")
    for swl in f:
        line = []
        words = swl.split()
        for sww in words:
            w = sww.translate(str.maketrans('', '', string.punctuation))
            w = w.lower()
            if (w not in obs_map):
                obs_map[w] = obs_counter
                obs_counter += 1
            line.append(obs_map[w])
        star_wars_lines.append(line)
        if obs_counter > 3000:
            break
    f.close()
    obs_map_r = {}
    for key in obs_map:
        obs_map_r[obs_map[key]] = key
    data = sonnets + star_wars_lines
    # Train HMM
    model = HMM.unsupervised_HMM(data, 5, 10)
    num_states = model.L
    # Print one blank line to make it pretty
    print("")
    # Generate sonnet
    last_state = np.random.choice(num_states, p=model.A_start)
    for n_lines in [4, 4, 2]:
        for l_no in range(n_lines):
            curr_state = last_state
            line = ""
            for _ in range(7):
                w, curr_state = add_word(curr_state, model.A, model.O)
                new_word = obs_map_r[w]
                line += new_word
                line += " "                
            last_state = curr_state
            print(line.capitalize())
        print("") 
        
hmm_shakespeare_sonnet_goal1()
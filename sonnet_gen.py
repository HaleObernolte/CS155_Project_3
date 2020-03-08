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
        count += syl_map(word)
    return count

# Returns a 10 syllable substring of a line, if it exists
def get_10_syls(line, syl_map):
    out = ""
    syls = 0
    words = line.split()
    for word in words:
        if (syls == 10):
            return out
        out += word
        out += " "
        if word not in syl_map:
            print("ERROR:", word, "not found in syllable map.")        
        syls += syl_map[word]
    return ""


###########################
# Shakespeare generation  #
###########################

# Generates one sheakespearian sonnet using a HMM
def hmm_shakespeare_sonnet():
    # Load in everything
    sonnets, obs_map = sp.get_sonnets("data/shakespeare.txt", 900)
    obs_map_r = {}
    for key in obs_map:
        obs_map_r[obs_map[key]] = key
    syl_map = sp.get_syllable_map("data/Syllable_dictionary.txt")
    # Train HMM
    model = HMM.unsupervised_HMM(sonnets, 10, 10)
    # Print one blank line to make it pretty
    print("")
    # Generate quatrain 1
    for _ in range(4):
        line = ""
        while (not line):
            l = ""
            emission, states = model.generate_emission(10)
            for e in emission:
                l += obs_map_r[e]
                l += " "
            line = get_10_syls(l, syl_map)
        print(line)
    print("")
    # Generate quatrain 2
    for _ in range(4):
        line = ""
        while (not line):
            l = ""
            emission, states = model.generate_emission(10)
            for e in emission:
                l += obs_map_r[e]
                l += " "
            line = get_10_syls(l, syl_map)
        print(line)
    print("")
    # Generate couplet
    for _ in range(2):
        line = ""
        while (not line):
            l = ""
            emission, states = model.generate_emission(10)
            for e in emission:
                l += obs_map_r[e]
                l += " "
            line = get_10_syls(l, syl_map)
        print(line)
        
hmm_shakespeare_sonnet()
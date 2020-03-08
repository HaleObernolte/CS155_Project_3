##############################################
# sonnet_processing.py                       #
#                                            #
# Code for processing data files of sonnets. #
#                                            #
# Team name: 0 Loss or Bust                  #
##############################################



############
# Imports  #
############
import string


#####################
# File processing   #
#####################

# Returns a list of all sonnets in file given by f_name.
#    In this case, each sonnet is just a list of words
def get_sonnets(f_name, max_words=4000):
    f = open(f_name, "r")
    # Skip first sonnet number
    f.readline()
    # Parse sonnets
    obs_counter = 0
    obs_map = {}    
    sonnets = []
    while (obs_counter < max_words):
        sonnet = []
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
            for word in words:
                w = word.translate(str.maketrans('', '', string.punctuation))
                w = w.lower()
                if w not in obs_map:
                    # Add unique words to the observations map.
                    obs_map[w] = obs_counter
                    obs_counter += 1
                # Add the encoded word.
                sonnet.append(obs_map[w])
        sonnets.append(sonnet)
        # Check for EOF
        if (not cont):
            break        
    return sonnets, obs_map


# Returns a map from words to their number of syllables
#
# Currently ignores words having different syllables when used at the end.
def get_syllable_map(f_name):
    f = open(f_name, "r")
    syl_map = {}
    for line in f:
        words = line.split()
        w = words[0]
        w = w.translate(str.maketrans('', '', string.punctuation))
        w = w.lower()
        n_str = words[-1]
        if "E" in n_str:
            n_str = words[-2]
        n = int(n_str)
        if w not in syl_map:
            syl_map[w] = n
    return syl_map
    
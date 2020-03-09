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

# Returns a list of all sonnets in file given by f_name, as well as the word map
#   and list of rhyme tuples.
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
        rhyme = ()
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
            for i in range(len(words)):
                word = words[i]
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


# Returns all rhyme pairs used by Shakespeare
#
# Each rhyme pair is repeated in the list whenever it is een so that taking a
# random rhyme from the list is sampling from Shakespeare's distribution.
# NOTE: Assumes you have deleted Sonnets 99 and 123
def get_rhymes(f_name):
    f = open(f_name, "r")
    rhymes = []
    line_no = 1
    a = ()
    b = ()
    g = ()
    for line in f:
        words = line.split()
        if (len(words) < 2):
            continue        
        w = words[-1]
        w = w.translate(str.maketrans('', '', string.punctuation))
        w = w.lower()
        # Add to correct tuple
        if (line_no == 1 or line_no == 3 or line_no == 5 or line_no == 7 or line_no == 9 or line_no == 11):
            a = a + (w,)
            if (len(a) == 2):
                rhymes.append(a)
                a = ()
        elif (line_no == 2 or line_no == 4 or line_no == 6 or line_no == 8 or line_no == 10 or line_no == 12):
            b = b + (w,)
            if (len(b) == 2):
                rhymes.append(b)
                b = ()
        elif (line_no == 13):
            g = g + (w,)
        else:
            g = g + (w,)
            rhymes.append(g)
            g = ()
            line_no = 0
        line_no += 1
    return rhymes
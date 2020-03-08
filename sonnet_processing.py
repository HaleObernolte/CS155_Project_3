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
def get_sonnets(f_name):
    f = open(f_name, "r")
    # Skip first sonnet number
    f.readline()
    # Parse sonnets
    obs_counter = 0
    obs_map = {}    
    sonnets = []
    while (True):
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
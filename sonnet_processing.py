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
                sonnet.append(w.lower())
        sonnets.append(sonnet)
        # Check for EOF
        if (not cont):
            break        
    return sonnets
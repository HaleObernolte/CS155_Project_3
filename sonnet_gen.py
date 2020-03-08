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


###########################
# Shakespeare generation  #
###########################

# Generates one sheakespearian sonnet using a HMM
def hmm_shakespeare_sonnet():
    sonnets, obs_map = sp.get_sonnets("data/shakespeare.txt")
    model = HMM.unsupervised_HMM(sonnets, 10, 100)
    # Generate quatrain 1
    for _ in range(4):
        # 8 words is close enough to 10 syllables, right?
        emission, states = HMM.generate_emission(8)
        line = ""
        for e in emission:
            line += obs_map[e]
            line += " "
        print(line)
        
hmm_shakespeare_sonnet()
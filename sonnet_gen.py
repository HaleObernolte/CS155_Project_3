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
    sonnets, obs_map = sp.get_sonnets("data/shakespeare.txt", 900)
    obs_map_r = {}
    for key in obs_map:
        obs_map_r[obs_map[key]] = key
    model = HMM.unsupervised_HMM(sonnets, 10, 10)
    # Generate quatrain 1
    for _ in range(4):
        # 8 words is close enough to 10 syllables, right?
        emission, states = model.generate_emission(8)
        line = ""
        for e in emission:
            line += obs_map_r[e]
            line += " "
        print(line)
    print("")
    # Generate quatrain 2
    for _ in range(4):
        # 8 words is close enough to 10 syllables, right?
        emission, states = model.generate_emission(8)
        line = ""
        for e in emission:
            line += obs_map_r[e]
            line += " "
        print(line)
    print("")
    # Generate couplet
    for _ in range(2):
        # 8 words is close enough to 10 syllables, right?
        emission, states = model.generate_emission(8)
        line = ""
        for e in emission:
            line += obs_map_r[e]
            line += " "
        print(line)
        
hmm_shakespeare_sonnet()
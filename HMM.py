########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.
        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.
            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.
        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        # Initialize first 'layer'
        for i in range(self.L):
            probs[1][i] = self.A_start[i] * self.O[i][x[0]]
            seqs[1][i] = str(i)
        
        # Run viterbi
        for level in range(2, M + 1):
            w = x[level - 1]
            for state in range(self.L):
                best_prob = -1
                best_state = -1
                for prev_state in range(self.L):
                    p = probs[level - 1][prev_state] * self.A[prev_state][state] * self.O[state][w]
                    if p > best_prob:
                        best_prob = p
                        best_state = prev_state
                probs[level][state] = best_prob
                seqs[level][state] = seqs[level - 1][best_state] + str(state)
        
        # Get best final sequnce
        best_prob = -1
        max_seq = "ERROR"
        for state in range(self.L):
            if probs[M][state] > best_prob:
                best_prob = probs[M][state]
                max_seq = seqs[M][state]
        return max_seq


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.
        Returns:
            alphas:     Vector of alphas.
                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.
                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # Initialize first layer
        for z in range(self.L):
            alphas[1][z] = self.O[z][x[0]] * self.A_start[z]
            
        # Run forward algorithm 
        for level in range(2, M + 1):
            w = x[level - 1]
            for z in range(self.L):
                p_sum = 0.0
                for prev_state in range(self.L):
                    p_sum += self.A[prev_state][z] * alphas[level - 1][prev_state]
                alphas[level][z] = self.O[z][w] * p_sum
                
        # Ignore first row
        new_alphas = []
        for i in range(1, len(alphas)):
            new_alphas.append(alphas[i])
        # Normalize, if necessary
        if (normalize):
            for w in range(M):
                p_sum = sum(new_alphas[w])
                for z in range(self.L):
                    if (p_sum > 0.0):
                        normed = new_alphas[w][z] / p_sum
                        new_alphas[w][z] = normed                        

        return new_alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
            normalize:  Whether to normalize each set of beta_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.
        Returns:
            betas:      Vector of betas.
                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.
                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # Initialize last layer 
        for z in range(self.L):
            betas[M][z] = 1
        
        # Run backward algorithm
        for level in range (M - 1, 0, -1):
            w = x[level]
            for z in range(self.L):
                p_sum = 0.0
                for j in range(self.L):
                    p_sum += betas[level + 1][j] * self.A[z][j] * self.O[j][w]
                betas[level][z] = p_sum
        
        # Ignore first row
        new_betas = []
        for i in range(1, len(betas)):
            new_betas.append(betas[i])
        # Normalize, if necessary
        if (normalize):
            for w in range(M):
                p_sum = sum(new_betas[w])
                for z in range(self.L):
                    if (p_sum > 0.0):
                        normed = new_betas[w][z] / p_sum
                        new_betas[w][z] = normed                        
                    
        return new_betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.
        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.
            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.
                        Note that the elements in X line up with those in Y.
        '''


        # Get total occurances of each y
        Y_tots = [0.0 for _ in range(self.L)]
        Y_tots_t = [0.0 for _ in range(self.L)]
        for ys in Y:
            for i in range(len(ys)):
                y = ys[i]
                Y_tots[y] += 1 
                if (i < len(ys) - 1):
                    Y_tots_t[y] += 1
        
        # Calculate each element of A using the M-step formulas.
            
        self.A = [[0. for _ in range(self.L)] for _ in range(self.L)]
        for ys in Y:
            for s in range(1, len(ys)):
                prev_state = ys[s - 1]
                state = ys[s]
                self.A[prev_state][state] += 1.0 / Y_tots_t[prev_state]            
            

        # Calculate each element of O using the M-step formulas.

        self.O = [[0. for _ in range(self.D)] for _ in range(self.L)]
        for t in range(len(X)):
            xs = X[t]
            ys = Y[t]
            for s in range(len(xs)):
                state = ys[s]
                w = xs[s]
                self.O[state][w] += 1.0 / Y_tots[state]            

        pass


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.
        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.
            N_iters:    The number of iterations to train on.
        '''

        # Matrices are already initialized
        for iter in range(N_iters):
            # Initialize temp matrices
            A_num = [[0.0 for _ in range(self.L)] for _ in range(self.L)]
            A_den = [[0.0 for _ in range(self.L)] for _ in range(self.L)]
            O_num = [[0.0 for _ in range(self.D)] for _ in range(self.L)]
            O_den = [[0.0 for _ in range(self.D)] for _ in range(self.L)]
            # Iterate over sequences
            for x in X:
                # Get alphas and betas
                alpha = self.forward(x, normalize=True)
                beta = self.backward(x, normalize=True)
                
                # Calculate gamma
                gamma = [[0.0 for _ in range(self.L)] for _ in range(len(x))]
                for t in range(len(x)):
                    t_sum = 0.0
                    for i in range(self.L):
                        g = alpha[t][i] * beta[t][i]
                        gamma[t][i] = g
                        t_sum += g
                    for i in range(self.L):
                        if (t_sum > 0.0):
                            gamma[t][i] /= t_sum
                # Calculate xi
                xi = [[[0.0 for _ in range(self.L)] for _ in range(self.L)] for _ in range(len(x) - 1)]
                for t in range(len(x) - 1):
                    t_sum = 0.0
                    for i in range(self.L):
                        for j in range(self.L):
                            p = alpha[t][i] * self.A[i][j] * beta[t + 1][j] * self.O[j][x[t + 1]]
                            xi[t][i][j] = p
                            t_sum += p
                    for i in range(self.L):
                        for j in range(self.L):
                            if (t_sum > 0.0):
                                xi[t][i][j] /= t_sum
                # Add temp matrix contributions
                for t in range(len(x)):
                    # A temp contributions
                    if (t < len(x) - 1):
                        for i in range(self.L):
                            for j in range(self.L):
                                A_num[i][j] += xi[t][i][j]
                                A_den[i][j] += gamma[t][i]
                    # O temp contributions
                    for i in range(self.L):
                        for w in range(self.D):
                            if (x[t] == w):
                                O_num[i][w] += gamma[t][i]
                            O_den[i][w] += gamma[t][i]
                        
                    
            # Update A and O matrices           
            for a in range(self.L):
                for b in range(self.L):
                    if (A_den[a][b] == 0.0):
                        self.A[a][b] = 0.0
                    else:
                        self.A[a][b] = A_num[a][b] / A_den[a][b]
            for a in range(self.L):
                for b in range(self.D):
                    if (O_den[a][b] == 0.0):
                        self.O[a][b] = 0.0
                    else:
                        self.O[a][b] = O_num[a][b] / O_den[a][b]
            
        pass


    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 
        Arguments:
            M:          Length of the emission to generate.
        Returns:
            emission:   The randomly generated emission as a list.
            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []

        curr_state_dist = self.A_start
        for _ in range(M):
            r = random.random()
            s = 0.0
            z = -1
            for i in range(len(curr_state_dist)):
                s += curr_state_dist[i]
                if r < s:
                    z = i
                    break
            if (z < 0):
                print("ERROR: No state selected in emission generation.")
                exit()
            states.append(z)
            r = random.random()
            s = 0.0
            for i in range(len(self.O[z])):
                s += self.O[z][i]
                if r < s:
                    emission.append(i)
                    break
            curr_state_dist = self.A[z]  
                

        return emission, states


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.
    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.
        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.
    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.
        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    random.seed(2020)
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    random.seed(155)
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
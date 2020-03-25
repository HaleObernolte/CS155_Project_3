#########################################
# RNN.py                         #
#                                       #
# Implements a RNN model for sonnets  #
#########################################

############
# Imports  #
############
import sonnet_processing as sp
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam
from keras.callbacks.callbacks import EarlyStopping
import random
import numpy as np
import sys


# Returns a list of all sonnets in file given by f_name
def get_char_sonnets(f_name, max_words=4000):
    ''' Based on a function written earlier. This '''
    ''' generates sonnets of just characters. This way, '''
    ''' the characters are used for predictions. '''
    f = open(f_name, "r")
    # Skip first sonnet number
    f.readline()
    # Parse sonnets
    obs_counter = 0
    sonnets = []
    while (obs_counter < max_words):
        sonnet = ''
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

                # Add the encoded word.
            sonnet += line
        sonnets.append(sonnet)
        # Check for EOF
        if (not cont):
            break
    return sonnets


def predictSonnet(allSonnetText, maxlen, chars, char_indices, indices_char, model):
    ''' Using the model and other parameters, this is an attempt to predict '''
    ''' a random sonnet. '''
    poem = ''

    sentence = 'Shall I compare thee to a summer\'s day?\n'
    lines = 1

    while lines < 14:
        x_pred = np.zeros((1, maxlen, len(chars)), dtype=np.bool)
        subsection = sentence[-40:]
        #print(subsection)
        #print(len(subsection))
        for t, char in enumerate(subsection):
            x_pred[0, t, char_indices[char]] = 1.
        probChars = model.predict(x_pred)[0]
        # maxProb = max(probChars)
        indexProb = sample(probChars, temperature=0.75)

        newchar = indices_char[indexProb]
        sentence = sentence + newchar
        if newchar == '\n':
            lines += 1

    print(sentence)
    #print(sentence[sentence.index('\n')+1:])
    # Draw softmax samples from trained model
    '''start_index = random.randint(0, len(allSonnetText) - maxlen - 1)
    print(maxlen)
    # New Poem
    poem = ''
    # Old sentence
    oldSentence = allSonnetText[start_index: start_index + maxlen]

    oldSentence = oldSentence[:maxlen]
    # Sentence being generated
    newSentence = oldSentence
    print(oldSentence)
    # We generate each line
    for j in range(14):
        oldSentence = newSentence
        newSentence = ''
        i = 0
        # Generate each character
        while i < maxlen:
            # Create an array of x's
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(oldSentence):
                if t < maxlen:
                    x_pred[0, t, char_indices[char]] = 1.
                else:
                    #print(t)
                    #print(char)
                    #print(oldSentence)
                    #print(len(oldSentence))
                    #print(newSentence)
                    #print("Why? Is this happening?")
                    continue

            # Predict on x's
            preds = model.predict(x_pred, verbose=0)[0]
            # diversity can be second arg for sample: 0.2, 0.5, 1, 1.2

            # Predict next sample
            next_index = sample(preds)
            next_char = indices_char[next_index]
            # Check what the character is, and add it if it is
            # a valid character
            #print("hello")
            # next char must be space only if i > 0
            # next char is not space or i > 0
            if next_char != '\n' and next_char != '$' and (next_char.isalnum() or i > 0):
                newSentence = newSentence + next_char
                #print(newSentence)
                i += 1
            # Break if newline
            if next_char == '\n':
                print(j)
                break
        # Get the list of the words, and remove the last partially
        # formed word.
        wordList = newSentence.split()
        #print(wordList)
        if len(wordList) > 1:
            wordList = wordList[:-1]
        #print("hey")
        #print(wordList)
        # Add to a poem as a string
        newSentence = ' '.join(wordList)
        #print("howdy")
        #print(newSentence)
        #newSentence = newSentence + '\n'
        #print("howdy1")
        #print(newSentence)
        poem += newSentence
        poem += "\n"
        #print("howdy2")
        #print(poem)'''
    #print(newSentence)
    #print("----------------------------------------------------")
    #print(poem)
    #print("----------------------------------------------------")



def rnn_shakespeare_sonnet():
    ''' Deal with the data and engine running the model. '''
    # Load in everything
    sonnetsAsNums, obs_map = sp.get_sonnets("data/shakespeare.txt", 900)
    #print(obs_map)
    obs_map_r = {}
    sonnets = get_char_sonnets("data/shakespeare.txt", max_words=900)
    for key in obs_map:
        obs_map_r[obs_map[key]] = key
    syl_map = sp.get_syllable_map("data/Syllable_dictionary.txt")
    # Train HMM
    RNN(sonnets, sonnetsAsNums, obs_map)


def RNN(sonnets, sonnetsAsNums, obs_map):
    ''' Runs the RNN model. '''
    # Note: ADJUST indicates parameters that can be adjusted

    sizeOfVocab = len(obs_map)
    # 1. PREPARE THE INPUT
    allSonnetText = ''
    endOfSonnetChar = '$'
    # Collect the sonnet text?
    for sonnet in sonnets:
        allSonnetText += sonnet[:-1] + endOfSonnetChar + sonnet[-1]
    # Retrieve all the characters in a set.
    chars = sorted(list(set(allSonnetText)))

    # encode the chars
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # Max length of the sequence, thought to be 40
    # CAN ADJUST
    maxlen = 40

    # relevant section of the problem:
    # Semi-redundant sequences (sequences starting every step_th char)
    # training data: sequences of fixed length from the corpus
    # take all possible subsequences of 40 consecutive chars from data set,
    # but for speed, using semi-redundant sequences

    # Train x is sequences of maxlen
    # ADJUST????
    step = 1
    sequences = []
    next_chars = []
    for sonnet in sonnets:
        for i in range(0, len(sonnet) - maxlen, step):
            sequences.append(list(sonnet[i: i + maxlen]))
            next_chars.append(sonnet[i + maxlen])
    # x: length(input, maximum length of sequence, length of all chars)
    #print(sequences[0])
    #print(len(sequences))
    #print(len(sequences[0]))
    #print(len(sequences[0][0]))
    #print(len(sequences[0][[0]]))
    #x = np.array(sequences)
    #print(x.ndim)
    #print(x)
    #x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    #print(x.shape)
    #print(len(x[0]))
    #x = np.reshape()
    #y = np.array(next_chars)
    #print(y.shape)
    # predict on 1, maxlen is 40, len(chars)
    x = np.zeros((len(sequences), maxlen, len(chars)), dtype=np.bool)
    #y: length(input, length of all chars)
    y = np.zeros((len(sequences), len(chars)), dtype=np.bool)
    for i, seq in enumerate(sequences):
        for t, char in enumerate(seq):
            #print(seq, ", ", str(i) + ", ", char, ", " + str(t))
            # ith sequence, tth char in sequence, the char's index in dict
            x[i, t, char_indices[char]] = 1
        # ith sequence, the char's index in dict
        #print(next_chars[i])
        #print(char_indices[next_chars[i]])
        y[i, char_indices[next_chars[i]]] = 1

    # 2. CREATE THE MODEL AND FIT THE DATA
    model = Sequential()
    # char based LSTM model, single layer of 100-200 LSTM
    # ADJUST: 100-200 LSTM
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    # fully connected dense output layer with softmax nonlinearity
    model.add(Dense(len(chars), activation='softmax'))
    print("optimizer: Adam eta=0.001, temp = 0.75")
    optimizer = Adam(learning_rate=0.001)
    # minimize categorical cross-entropy
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    # train for sufficient number of epochs to converge loss try different numbers,
    # graph
    # ADJUST EPOCHS
    # POSSIBLY ADJUST batch_size
    es = EarlyStopping(monitor='loss')
    model.fit(x, y, batch_size=40, epochs=500, callbacks=[es])

    #print(model.predict(x_pred))
    # 3. EXAMPLE 1 WHERE WE TRY TO GENERATE 40 CHARS
    # 40 random characters

    # TODO: Not quite sure how to do this
    # You can see the method
    predictedPoem = predictSonnet(allSonnetText, maxlen, chars, char_indices, indices_char, model)
    return predictedPoem


# https://keras.io/examples/lstm_text_generation/
def sample(preds, temperature=1.0):
    ''' Sample function from a probability array '''
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    for i in range(len(preds)):
        if preds[i] > 0:
            preds[i] = np.log(preds[i])
        else:
            preds[i] = 0
    preds = preds / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


rnn_shakespeare_sonnet()
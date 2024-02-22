import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    hmm = HiddenMarkovModel(**mini_hmm)


    # bad seq check
    bad_sequence = ['sunny', '--APPLE--', 'rainy']
    with pytest.raises(ValueError):
        hmm.forward( np.array( bad_sequence ) )
        hmm.viterbi( np.array( bad_sequence ) )


    # empty input check
    empty = np.empty(0)
    assert hmm.forward(empty) == 0
    assert hmm.viterbi(empty) == []


    # forward test
    test_forward = hmm.forward(mini_input['observation_state_sequence'])
    assert round(test_forward, 3) == 0.035


    # viterbi test
    test_bestpath = hmm.viterbi(mini_input['observation_state_sequence'])
    assert test_bestpath == list( mini_input['best_hidden_state_sequence'] )



def test_full_weather():

    """
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """
    full_hmm=np.load('./data/full_weather_hmm.npz')
    full_input=np.load('./data/full_weather_sequences.npz')

    hmm = HiddenMarkovModel(**full_hmm)

    # viterbi test
    test_bestpath = hmm.viterbi(full_input['observation_state_sequence'])
    assert test_bestpath == list( full_input['best_hidden_state_sequence'] )


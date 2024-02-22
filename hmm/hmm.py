import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
        ''' Edge case detection '''
        # empty input check
        if len(input_observation_states) == 0:
            return 0.0
        
        # unknown observation check
        for obs in input_observation_states:
            if obs not in self.observation_states_dict:
                raise ValueError(f"Unknown observation '{obs}' in input sequence!")


        # Step 1. Initialize variables
        N = len(self.hidden_states)
        T = len(input_observation_states)
        
        # Observation vs Time
        forward_prob = np.zeros((N, T))


        # Step 2. Calculate probabilities

        # initialization
        first_obs_index = self.observation_states_dict[input_observation_states[0]]
        
        # pi * b(o1) for all hidden states to o1
        forward_prob[ : , 0] = self.prior_p * self.emission_p[ :, first_obs_index]


        # recursion
        # skip T=1 obs given init
        for t in range(1, T): 
            obs_index = self.observation_states_dict[input_observation_states[t]]

            for j in range(N):
                # (forward_prob[i, t-1]) * (forward_prob[i, t-1] --> hidden[j]) * (hidden[j] --> observe[t])
                # dot should yeild a vectorized way of doing it over all i's
                forward_prob[j, t] = np.dot(forward_prob[ : , t-1], self.transition_p[ : , j]) * self.emission_p[j, obs_index]


        # Step 3. Return final probability 
        
        # final prob should be the sum of all diffrent j's that got us to last step
        forward_probability = np.sum( forward_prob[ : , -1 ] )

        return forward_probability


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        
        ''' Edge case detection '''
        # empty input check
        if len(decode_observation_states) == 0:
            return []
        
        # unknown observation check
        for obs in decode_observation_states:
            if obs not in self.observation_states_dict:
                raise ValueError(f"Unknown observation '{obs}' in input sequence!")


        # Step 1. Initialize variables
        N = len(self.hidden_states)
        T = len(decode_observation_states)
        
        # Observation vs Time, + tracking for traceback
        viterbi_table = np.zeros((N, T))
        backpointer = np.zeros((N, T), dtype=int)


        # Step 2. Calculate probabilities

        # initialization
        first_obs_index = self.observation_states_dict[decode_observation_states[0]]
        
        # pi * b(o1) for all hidden states to o1
        viterbi_table[ : , 0] = self.prior_p * self.emission_p[ :, first_obs_index]

        # recursion
        # skip T=1 obs given init
        for t in range(1, T): 
            obs_index = self.observation_states_dict[decode_observation_states[t]]

            for j in range(N):
                #      (forward_prob[i, t-1]) * (forward_prob[i, t-1] --> hidden[j]) * (hidden[j] --> observe[t])
                prob = viterbi_table[ : , t-1] * self.transition_p[ : , j] * self.emission_p[j, obs_index]
                
                viterbi_table[j, t] = np.max(prob)
                backpointer[j, t] = np.argmax(prob)
        

        # Step 3. Traceback
        best_last_state = np.argmax(viterbi_table[:, T-1])
        bestpath = [ best_last_state ]

        # go backwards from backpointer, insert best states at the beg (index 0)
        for t in range(T-1, 0, -1):
            bestpath.insert(0, backpointer[bestpath[0], t])


        # Step 4. Return best hidden state sequence 
        best_hidden_state_sequence = [ self.hidden_states_dict[state] for state in bestpath ]

        return best_hidden_state_sequence
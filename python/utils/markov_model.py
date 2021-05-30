import random

import numpy as np
import pandas as pd
from numpy.linalg import matrix_power
from functools import partial

from python.utils.timer import Stage

# rand_data = [random.choice(list(Stage)) for i in range(100)]


def generate_transition_matrix(data):
    """ Generate the Transition Matrix
    Uses a probabilistic method to generate transition matrix. Computes all transitions from one state
    to another then use the transition to generate transition Matrix
    """

    _df = {}
    # Current State data
    _df['state'] = data[:-1]
    # Next state of each current state is the next data in the list
    _df ["next_state"] = data[1:]
    cleaned_data = pd.DataFrame(_df)

    transitions = {d: {} for d in list(Stage)}
    # Check for transitions between states and store count
    for i in transitions.keys():
        for j in transitions.keys():
            transitions[i][j] = cleaned_data[
                    (cleaned_data["state"] == i) & (cleaned_data["next_state"] == j)].shape[0]

    # Calculate the Probability of Transition based on data from transtions
    df = pd.DataFrame(transitions)
    for i in range(df.shape[0]):
        df.iloc[i] = df.iloc[i]/(df.iloc[i].sum() or 1)

    transition_matrix = df.values
    return transition_matrix

class MarkovChain:
    """ The Markov Chain Model Predictor """

    def __init__(self):
        """ Initialize the necessary Variables
        data is a 1D array containing states in chronological order which they were transitioned to
        """

        # Store the transitions from a state to this dict
        self.transition_matrix = None

    def get_current_state_vector(self, current_state):
        """ Get the current state vector
        This is usually of the form [0, 0, 1, 0], where 1 indicates the current state, and other states
        are in 0 state
        """

        v_state = np.zeros([len(list(Stage))])
        v_state[current_state.value] = 1
        return v_state

    def next_state(self, current_state, time_step=1, transition_matrix=None):
        """
        Returns the state of the random variable at the next time
        instance.

        Parameters
        ----------
        current_state: str
            The current state of the system.
        """
        tmx = np.atleast_2d(transition_matrix)

        current_state_vector = self.get_current_state_vector(current_state)
        a = current_state_vector.dot(matrix_power(tmx, time_step))
        return list(Stage)[np.argmax(a)]

    def predict(self, current_state, transition_matrix=None, no_predictions=1):
        """
        Generates the next states of the system.

        Parameters
        ----------
        current_state: str
            The state of the current random variable.

        no: int
            The number of future states to generate.
        """
        future_states = []
        for i in range(no_predictions):
            next_state = self.next_state(current_state, time_step=i+1, transition_matrix=transition_matrix)
            future_states.append(next_state)
        return future_states

import numpy as np
#### MONTE CARLO #####

def play_episode_MC(gw, policy, gamma, ending_states = [0]):
    state = np.random.randint(gw.get_state_count())
    states_and_rewards = [(state, 0)]
    G = 0
    while True:
        actions = gw.get_available_actions(state)
        action = actions[np.random.choice(len(actions), p=policy)]
        transitions = gw.get_transitions(state=state, action=action)
        trans_probs = []
        for _, _, probabiliity in transitions:
            trans_probs.append(probabiliity)
        next_state, reward, _ = transitions[np.random.choice(len(trans_probs), p=trans_probs)]
        states_and_rewards.append((next_state, reward))
        state = next_state
        if state in ending_states:
            break
    G = 0
    states_and_returns = []
    first = True
    for s, r in reversed(states_and_rewards):
        states_and_returns.append((s, G))
        G = r + gamma*G
    states_and_returns.reverse()
    return states_and_returns

def policy_eval_MC(gw, policy, gamma, episodes = 1000, rseed = 33, ending_states = [0], V = None):
    returns = {}
    n_states = gw.get_state_count()
    if V is None:
        V = np.zeros(n_states)
    else:
        V = V.copy()
    for s in range(n_states):
        returns[s] = []
    np.random.seed(rseed)
    for t in range(episodes):
        # Devuleve cada estado con su Value para ese episodio
        states_and_returns = play_episode_MC(gw, policy, gamma, ending_states)
        seen_state = set()
        for s, G in states_and_returns:
            if s not in seen_state: # Esta linea se puede comentar y "da lo mismo" en el limite
                returns[s].append(G)
                seen_state.add(s)
    for s, Gs in returns.items():
        if len(Gs)>0:
            V[s] = np.mean(Gs)
        else:
            V[s] = 0
    return V


### TD0 ###
def play_episode_TD0(V, gw, policy, gamma, alpha = 1, ending_states = [0]):
    state = np.random.randint(gw.get_state_count())
    G = 0
    while True:
        actions = gw.get_available_actions(state)
        action = actions[np.random.choice(len(actions), p=policy)]
        transitions = gw.get_transitions(state=state, action=action)
        trans_probs = []
        for _, _, probabiliity in transitions:
            trans_probs.append(probabiliity)
        next_state, reward, _ = transitions[np.random.choice(len(trans_probs), p=trans_probs)]
        V[state] = V[state] + alpha*(reward  +  gamma * V[next_state] - V[state]) # error = reward  +  gamma * V[next_state] - V[s]
        state = next_state
        if state in ending_states:
            break
    return V

def policy_evel_TD0(gw, policy, gamma, alpha = 1, episodes=1000, ending_states = [0], V = None):
    if V is None:
        V = np.zeros(gw.get_state_count())
    else:
        V = V.copy()
    for i in range(episodes):
        V = play_episode_TD0(V, gw, policy, gamma, alpha, ending_states)
    return V

### Dynamic Programming ###
def get_equal_policy(state):
    # build a simple policy where all 4 actions have the same probability, ignoring the specified state
    policy = (("up", .25), ("right", .25), ("down", .25), ("left", .25))
    return policy

def policy_eval_two_arrays(state_count, gamma, theta, get_policy, get_transitions):
    """
    This function uses the two-array approach to evaluate the specified policy for the specified MDP:
    
    'state_count' is the total number of states in the MDP. States are represented as 0-relative numbers.
    
    'gamma' is the MDP discount factor for rewards.
    
    'theta' is the small number threshold to signal convergence of the value function (see Iterative Policy Evaluation algorithm).
    
    'get_policy' is the stochastic policy function - it takes a state parameter and returns list of tuples, 
        where each tuple is of the form: (action, probability).  It represents the policy being evaluated.
        
    'get_transitions' is the state/reward transiton function.  It accepts two parameters, state and action, and returns
        a list of tuples, where each tuple is of the form: (next_state, reward, probabiliity).  
        
    """
    V = state_count*[0]
    V_old = state_count*[0]
    #
    # INSERT CODE HERE to evaluate the policy using the 2 array approach 
    #
    delta = theta + 1
    iterations = 0
    while theta<delta:
        iterations += 1
        for state in range(state_count):
            outer_sum = 0
            for action, prob in get_policy(state):
                transitions = get_transitions(state=state, action=action)
                inner_sum = 0
                for next_state, reward, probabiliity in transitions:
                    inner_sum = inner_sum + probabiliity * (reward  +  gamma * V_old[next_state])
                outer_sum = outer_sum + prob * inner_sum
            V[state] = outer_sum
        delta = np.max(np.abs(np.array(V) - np.array(V_old)))
        V_old = V.copy()
    print('numero de iteraciones:', iterations)
    return V
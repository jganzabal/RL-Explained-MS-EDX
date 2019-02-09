n_states = 7
def set_states(n=7):
    global n_states
    n_states = n
    
def get_state_count():
    return n_states

def get_available_actions(state):
    actions = ["left", "right"]
    return actions

def get_transitions(state, action):
    global n_states
    if state in [0, n_states-1]:
        return [(state, 0, 1)]
    
    if action == "right":
        next_state_num = state + 1
    if action == "left":
        next_state_num = state - 1
        
    if next_state_num == n_states-1:
        return [(next_state_num, 1, 1)]
    else:
        return [(next_state_num, 0, 1)]

from copy import deepcopy
import numpy as np
import random

# Representación de los objetos del tablero
ZOMBIE = "z"
CAR = "c"
ICE_CREAM = "i"
EMPTY = "*"

# Acciones del carro
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTIONS = [UP, DOWN, LEFT, RIGHT]

# Clase del estado (tablero, posicion del carro)
class State:
    
    def __init__(self, grid, car_pos):
        self.grid = grid
        self.car_pos = car_pos
        
    def __eq__(self, other):
        return isinstance(other, State) and self.grid == other.grid and self.car_pos == other.car_pos
    
    def __hash__(self):
        return hash(str(self.grid) + str(self.car_pos))
    
    def __str__(self):
        return f"State(grid={self.grid}, car_pos={self.car_pos})"

# Funcion que ejecuta la accion indicada y retorna el nuevo estado
def act(state, action):
    
    def new_car_pos(state, action):
        p = deepcopy(state.car_pos)
        if action == UP:
            p[0] = max(0, p[0] - 1)
        elif action == DOWN:
            p[0] = min(len(state.grid) - 1, p[0] + 1)
        elif action == LEFT:
            p[1] = max(0, p[1] - 1)
        elif action == RIGHT:
            p[1] = min(len(state.grid[0]) - 1, p[1] + 1)
        else:
            raise ValueError(f"Unknown action {action}")
        return p
            
    p = new_car_pos(state, action)
    grid_item = state.grid[p[0]][p[1]]
    
    new_grid = deepcopy(state.grid)
    
    old = state.car_pos

    if grid_item == ZOMBIE:
        reward = -100
        is_done = True
        new_grid[old[0]][old[1]] = EMPTY
        new_grid[p[0]][p[1]] += CAR
    elif grid_item == ICE_CREAM:
        reward = 1000
        is_done = True
        new_grid[old[0]][old[1]] = EMPTY
        new_grid[p[0]][p[1]] += CAR
    elif grid_item == EMPTY:
        reward = -1
        is_done = False
        new_grid[old[0]][old[1]] = EMPTY
        new_grid[p[0]][p[1]] = CAR
    elif grid_item == CAR:
        reward = -1
        is_done = False
    else:
        raise ValueError(f"Unknown grid item {grid_item}")
    
    return State(grid=new_grid, car_pos=p), reward, is_done

# Funcion que devuelve el vector q de un estado
def q(state, action=None):
    
    if state not in q_table:
        q_table[state] = np.zeros(len(ACTIONS))
        
    if action is None:
        return q_table[state]
    
    return q_table[state][action]

# Elegir una accion
def choose_action(state):
    if random.uniform(0, 1) < eps:
        return random.choice(ACTIONS) 
    else:
        return np.argmax(q(state))




### Sobre este ejemplo ###

# Estancia del tablero
grid = [
    [ICE_CREAM, EMPTY, EMPTY],
    [EMPTY, EMPTY, EMPTY],
    [ZOMBIE, CAR, EMPTY]
]

# Estado inicial
start_state = State(grid=grid, car_pos=[2, 1])

random.seed(1) # for reproducibility

N_EPISODES = 20

MAX_EPISODE_STEPS = 100

MIN_ALPHA = 0.02

alphas = np.linspace(1.0, MIN_ALPHA, N_EPISODES)
gamma = 1.0
eps = 0.2

q_table = dict()

# Entrenamiento
for e in range(N_EPISODES):
    
    state = start_state
    total_reward = 0
    alpha = alphas[e]
    
    for _ in range(MAX_EPISODE_STEPS):
        action = choose_action(state)
        next_state, reward, done = act(state, action)
        total_reward += reward
        
        q(state)[action] = q(state, action) + alpha * (reward + gamma *  np.max(q(next_state)) - q(state, action))
        state = next_state
        if done:
            break
    print(f"Episode {e + 1}: total reward -> {total_reward}")


# Ejecución
r = q(start_state)
done = False
total_reward = 0

print("")
for row in grid:
    print(' '.join(row))
print("Recompesa total: " + str(total_reward))
print("Q: " + f"up={r[UP]}, down={r[DOWN]}, left={r[LEFT]}, right={r[RIGHT]}")
print("")

while (done == False):

    action = int(input("Digite la accion a realizar(UP=0, DOWN=1, LEFT=2, RIGHT=3) : "))

    start_state, reward, done = act(start_state, action)
    total_reward += reward
    r = q(start_state)

    for row in start_state.grid:
        print(' '.join(row))
    print("Recompesa total: " + str(total_reward))
    print("Q: " + f"up={r[UP]}, down={r[DOWN]}, left={r[LEFT]}, right={r[RIGHT]}")
    print("")

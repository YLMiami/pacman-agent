# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random, sys
import time
sys.path.append('./pacman-contest/src/contest')
import util as util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point, Stack

from queue import PriorityQueue
from itertools import product, count

#################
# Team creation #
#################

help_me = False

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########
    
### Utility Functions ###

def nearest_open_space(game_state, food):
    """
    Returns the closest cell with at least three free walls around the given food position.
    """
    stack = Stack()
    stack.push(food)
    visited = {food}

    while not stack.is_empty():
        cell = stack.pop()
        if not surrounded_by_walls(game_state, cell):
            return cell
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbour_cell = (cell[0] + dx, cell[1] + dy)
            if not is_wall(game_state, neighbour_cell) and neighbour_cell not in visited:
                stack.push(neighbour_cell)
                visited.add(neighbour_cell)

        if stack.is_empty():
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                neighbour_cell = (cell[0] + dx, cell[1] + dy)
                if not is_wall(game_state, neighbour_cell) and neighbour_cell not in visited:
                    stack.push(neighbour_cell)
                    visited.add(neighbour_cell)
    
    return food
    

def surrounded_by_walls(game_state, cell):
    """
    Checks if a cell is surrounded by walls on at least two sides.

    Args:
        game_state: The current game state.
        position (tuple): The position to check.

    Returns:
        bool: True if surrounded by walls, False otherwise.
    """
    wall_count = sum(
        is_wall(game_state, (cell[0] + dx, cell[1] + dy))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
    )
    return wall_count > 1

def is_wall(game_state, cell):
    """
    Returns True if the given cell is a wall.
    """
    wall_map = game_state.get_walls()
    x, y = (int(x) for x in cell)
    return wall_map[x][y]

def arena_width(game_state):
    """
    Returns the width of the game arena.
    """
    return game_state.get_walls().width

def arena_height(game_state):
    """
    Returns the heigth of the game arena.
    """
    return game_state.get_walls().height


### Reflex Agent ###

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.001):
        super().__init__(index, time_for_computing)
        self.start = None
        self.returning_home = False
        self.escape_deadlock = 0
        self.escape_deadlock_cell = [0, 0]

        # Defensive agent variables
        self.total_food_enemies = 0
        self.our_food = set()
        
        self.last_food_eaten_by_enemy = None
        self.previous_score = 0
        self.enemy_is_attacking = False
        self.steps_since_last_food_eaten = 0
         
        self.save_my_location = [(0, 0) for _ in range(15)]
        
        self.capsule = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)  # Start position of the agent (int x, int y)
        CaptureAgent.register_initial_state(self, game_state)
        self.total_food_enemies = self.get_total_food_enemies(game_state)
        self.our_food = set(self.get_our_food(game_state))
        
    ### Core Utilities ###

    def get_next_game_state(self, game_state, action):
        """
        Movement can occur in increments smaller than a grid cell, so this ensures 
        the state is only considered when the agent reaches a valid grid position.
        """
        # Resulting game state after the agent performs the action.
        next_game_state = game_state.generate_successor(self.index, action)
        pos = next_game_state.get_agent_state(self.index).get_position()
        # if pos is not a valid grid position (i.e. in the middle of a cell), return the next game state
        if pos != nearest_point(pos):
            return next_game_state.generate_successor(self.index, action)
        return next_game_state

    
    ### Position and Distance Calculations ###

    def get_my_position(self, game_state):
        """
        Returns the agent's current position.
        """
        my_loc = game_state.get_agent_state(self.index).get_position()
        return tuple(map(int, my_loc))
    
    def get_distances_to(self, game_state, positions):
        """
        Calculates maze distances from the agent's position to a list of target positions.
        """
        return [self.get_maze_distance(self.get_my_position(game_state), pos) for pos in positions]
    
    def calculate_distances_with_positions(self, game_state, positions):
        """
        Returns a list of tuples (distance, position) for given target positions.
        """
        return list(zip(self.get_distances_to(game_state, positions), positions)) 
    
    def sort_positions_by_distance(self, game_state, targets):
        """
        Sorts a list of (distance, position) tuples by distance and returns the positions.
        """
        distances_w_position_list = self.calculate_distances_with_positions(game_state, targets)
        return [d[1] for d in sorted(distances_w_position_list)]
    
    ### Game Information Helpers ###

    ## On Food ##
    def get_food_positions(self, game_state):
        """
        Returns a list of positions where food in the opponent's territory is located. Bon Appetit!
        Args:
            game_state: The current game state.

        Returns:
            list: list of (x, y) positions.
        """
        return self.get_food(game_state).as_list()
    
    def get_food_count(self, game_state):
        """
        Returns the count of remaining food items.
        """
        return len(self.get_food_positions(game_state))

    def num_carrying(self, game_state):
        """
        Returns the number of food items the agent is carrying.
        """
        return game_state.get_agent_state(self.index).num_carrying
    
    def get_edibles(self, game_state):
        """
        An edible is:
        - Food that the agent can eat (as returned by get_food_positions).
        -  Enemy Pacmen that the agent can "eat" (capture) when the agent is in ghost mode and not scared.
        Returns: 
            list: a sorted list of edibles positions [closest to the agent, ... farthest].
        """
        edibles = self.get_food_positions(game_state)

        # Add enemy Pacmen positions if the agent isn't scared
        if not self.scared(game_state):
            _, scared_ghosts = self.get_enemy_ghost_positions(game_state)
            edibles.extend(scared_ghosts)

        return self.sort_positions_by_distance(game_state, edibles)
    
    def get_our_food(self, game_state):
        """
        Returns a list of food positions in the agent's territory.
        """
        return self.get_food_you_are_defending(game_state).as_list()

    def get_total_food_enemies(self, game_state):
        """
        Returns the total number of food enemies can eat in our territory.
        """
        return len(self.get_our_food(game_state))
    

    def get_closest_foods_from_enemy_position(self, game_state, enemy):
        """
        Returns the closest food positions to enemy positions.

        Args:
            game_state: The current game state.
            enemy: The position of the enemy (x,y).
        
        Returns:
            list: list of (x, y) positions sorted by distance.
        """
        food_positions = self.get_our_food(game_state)
        closest_foods = sorted(food_positions, key=lambda food: self.get_maze_distance(food, enemy))
        return closest_foods

    
    ## On Pacman ##
    def is_pacman(self, game_state):
        """
        Checks if the agent is in Pacman mode (on the opponent's side).
        """
        return game_state.get_agent_state(self.index).is_pacman
    
    def get_enemy_states(self, game_state):
        """
        Returns the states of all enemy agents.
        """
        return [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]

    def get_enemy_positions(self, game_state, filter_fn=None):
        """
        Returns positions of enemy agents that satisfy the filter function.
        
        Args:
            filter_fn (function): A function to filter enemy states (default is None, meaning no filtering).
            
        Returns:
            List of positions (x, y) of enemy agents matching the filter.
        """
        enemy_states = self.get_enemy_states(game_state)
        return [tuple(map(int, pos)) for a in enemy_states if (pos := a.get_position()) is not None and (filter_fn is None or filter_fn(a))]

    def get_enemy_pacmen_positions(self, game_state):
        """
        Returns the positions of enemy Pacmen visible to the agent. 
        This means that they are on the agent side and can collect food in its territory.
        """
        return self.get_enemy_positions(game_state, filter_fn=lambda a: a.is_pacman)
    
    def get_enemy_pacmen_agent_can_see(self, game_state, agent_location):
        """
        Returns the positions of enemy attackers.
        """
        enemy_pacmen = self.get_enemy_pacmen_positions(game_state)
        return [p for p in enemy_pacmen if self.get_maze_distance(agent_location, p) < 5]


    def get_enemy_ghost_positions(self, game_state):
        """
        Returns the positions of visible enemy ghosts.
        An enemy is a **REGULAR** ghost when:
            - It is on its own side of the map (defensive mode).
            - It has no active scared timer (i.e., it has not been affected by a power pellet).
            Regular ghosts can kill Pacman agents that enter their territory.
        
        An enemy is a **SCARED** ghost when:
            - It is on its own side of the map (defensive mode).
            - It has an active scared timer due to a Pacman agent eating a power pellet.
            Scared ghosts can be eaten by Pacman agents.
        """
        # Ghosts that are not Pacman and have a scared_timer > 5
        scared_ghosts = self.get_enemy_positions(
            game_state,
            filter_fn=lambda a: not a.is_pacman and a.scared_timer > 5
        )

        # Ghosts that are not Pacman and have a scared_timer <= 5
        regular_ghosts = self.get_enemy_positions(
            game_state,
            filter_fn=lambda a: not a.is_pacman and a.scared_timer <= 5
        )

        return regular_ghosts, scared_ghosts

    def get_enemy_ghosts_agent_can_see(self, game_state, agent_location):
        """
        Returns the positions of the enemy ghosts the agent can see sorted by closeness.
        """
        regular_ghosts, scared_ghosts = self.get_enemy_ghost_positions(game_state)
        visible_ghosts = [g for g in regular_ghosts if self.get_maze_distance(agent_location, g) < 5]
        return sorted(visible_ghosts, key = lambda ghost: self.get_maze_distance(agent_location, ghost)), scared_ghosts

    def scared(self, game_state):
        """
        Returns True if the agent is currently scared.
            - Only agents in ghost mode can be scared.
            - Pacmen cannot be scared because they are in offensive mode when on the opponent's side, and power pellets do not affect them.
        """
        return game_state.get_agent_state(self.index).scared_timer != 0

    ### Strategic Helpers ###
    
    def get_edge_home_cells(self, game_state, enemy_home=False, get_idx=None):
        """
        Returns the edge cells (boundary cells) on either the agent's home or enemy's side.
        
        Args:
            enemy_home (bool): If True, calculates for the enemy side; otherwise, for the agent's side.
        
        Returns:
            List of valid (non-wall) edge cell positions.
        """
        mid_col = arena_width(game_state) // 2
        home_red, home_blue = mid_col - 1, mid_col
        edge_col = home_red if (not enemy_home + self.red) % 2 == 0 else home_blue
        if get_idx:
            edge_col = get_idx
        return [(edge_col, row) for row in range(1, arena_height(game_state) - 1) if not is_wall(game_state, (edge_col, row))]
    
    def get_edge_bottom_up_cells(self, game_state):
        """
        Returns the edge cells (boundary cells) on either the agent's home or enemy's side.
        
        Args:
            enemy_home (bool): If True, calculates for the enemy side; otherwise, for the agent's side.
        
        Returns:
            List of valid (non-wall) edge cell positions.
        """
        h = arena_height(game_state) - 1
        blue, red = range(arena_width(game_state) // 4, arena_width(game_state) // 2), range(arena_width(game_state) // 2 + 1, 3*arena_width(game_state) // 4)
        d = red if self.red else blue
        return [(col, row) for col in d for row in (1, h) if not is_wall(game_state, (col, row))]


    
    def get_closest_home_cell_position(self, game_state):
        """
        Finds the closest edge cell on the agent's home side for safe return.
        
        Returns:
            Tuple: The closest escape position (x, y).
        """
        home_list = self.get_edge_home_cells(game_state)
        home = self.sort_positions_by_distance(game_state, home_list)
        return home[0] # home[0] why not this?
    

    def get_closest_reachable_home_cell_position(self, game_state, agent_location=None, enemy_location=None):
        """
        Finds the closest edge cell on the agent's home side for safe return.
        
        Returns:
            Tuple: The closest escape position (x, y).
        """
        home_list = self.get_edge_home_cells(game_state)
        closest_homes = self.sort_positions_by_distance(game_state, home_list)

        for home_cell in closest_homes:
            distance_between_pacman_and_home_cell = self.get_maze_distance(agent_location, home_cell)
            distance_between_enemy_and_home_cell = self.get_maze_distance(enemy_location, home_cell)

            # Go home only if you can manage to reach it before the ghost catches you
            if distance_between_pacman_and_home_cell < distance_between_enemy_and_home_cell:
                print("Closest home cell:", home_cell)
                return home_cell
        print("No reachable home cell.")
        return None
    

    def get_farthest_home_cell_position(self, game_state):
        """
        Returns the farthest escape position when returning home. (?why farthest?)
        """
        edge_home_cells = self.get_edge_home_cells(game_state)
        edge_home_cells_distances = self.sort_positions_by_distance(game_state, edge_home_cells)
        return edge_home_cells_distances[-1]

    ### Pathfinding and Scoring ###

    def score(self, my_pos, target, distance_traveled, enemy_penalty):
        """
        Calculates a heuristic score for A* pathfinding with path-specific enemy avoidance.
        Adds a penalty if enemies are near or along the evaluated path to the target.
        """
        # Base score: f(n) = g(n) + h(n)
        base_score = distance_traveled + self.get_maze_distance(my_pos, target) + enemy_penalty
        #print("Base score:", base_score, my_pos, target, distance_traveled, enemy_penalty)      

        return base_score


    def Astar(self, game_state, target, excluded_positions=[]):

        pq = PriorityQueue()
        
        # Needed to resolve order in the priority queue if two states have the same score
        counter = count()

        # Initialize the priority queue
        self.add_to_prio_queue(pq, game_state, 0, None, None, target, counter)
        
        # Positions that we don't want to visit are treated as visited
        visited = set(excluded_positions)
        visited.add(self.start)
        
        # Early exit if already at the target
        if self.get_my_position(game_state) == target:
            return "Stop"

        while not pq.empty():
            current_state = self.get_from_prio_queue(pq)
            current_game_state, distance_traveled = current_state["game_state"], current_state["distance_traveled"]
            my_pos = self.get_my_position(current_game_state)

            if my_pos == target:
                break

            actions = current_game_state.get_legal_actions(self.index)
            for action in actions:
                next_game_state = self.get_next_game_state(current_game_state, action)
                my_pos = self.get_my_position(next_game_state)
                if my_pos not in visited:
                    visited.add(my_pos)
                    self.add_to_prio_queue(pq, next_game_state, distance_traveled + 1, current_state, action, target, counter)

        return self.first_action(current_state)
    
    def add_to_prio_queue(self, pq, game_state, distance_traveled, previous, action, target, counter):
        next_state = {
            "game_state" : game_state,
            "distance_traveled" : distance_traveled,
            "previous" : previous,
            "action" : action
        }
        # Simulate the next path positions (evaluate positions after this action)
        next_my_pos = self.get_my_position(game_state)
        enemy_penalty = self.simulate_next_positions(game_state, next_my_pos) if action else 0

        # Calculate heuristic score
        score = self.score(next_my_pos, target, distance_traveled, enemy_penalty)
        pq.put((score, next(counter), next_state))

    def simulate_next_positions(self, next_game_state, my_pos):
        """
        Simulates the next positions based on the agent's current position and an action.
        """

        # Evaluate the new position after taking the action
        # Add penalty if an enemy is near or on the path
        if next_game_state:
            enemy_ghosts, _ = self.get_enemy_ghosts_agent_can_see(next_game_state, my_pos)
            # print("Enemy ghosts:", enemy_ghosts)
            penalty = 0
            for ghost in enemy_ghosts:
                distance_to_position = self.get_maze_distance(ghost, my_pos)
                if distance_to_position < 2:  # Penalize if ghost is directly on or adjacent to the path
                    penalty += 50  # Apply a fixed penalty
            
        return penalty


    def get_from_prio_queue(self, pq):
        return pq.get()[2]
    
    def first_action(self, previous):
        """
        The way it's organized, the previous variable holds the last state that was visited. 

        Since the value of "previous" and of "action" in the start state is None, 
        we don't have to go till the last state but till the second to last state, 
        which holds the first action that was added in the priority queue.

        Example:

        {game_state:g, distance_traveled:1, previous:{game_state:g, distance_traveled:0, previous:None, action:None}, action:a}

        """
        while previous["previous"]["previous"]:
            previous = previous["previous"]
        return previous["action"]


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    Cool Attacker
    """
    FOOD_CARRYING_THRESHOLD = 5
    TIME_MARGIN_FACTOR = 4
    def choose_action(self, game_state):
        """
        Chooses the best action based on the agent's current state and context.
        """
        print("-" * 50)
        print("Attacker")

        # Check if the agent is Pacman (on opponent's side).
        if self.is_pacman(game_state):
            print("I'm Pacman.")
            return self.offensive_action(game_state)
        else:
            # Agent is a ghost (on its own side).
            print("I'm a ghost.")
            return self.defensive_action(game_state)

    def offensive_action(self, game_state):
        """
        Determines the best offensive action when the agent is Pacman.
        """
        agent_location = self.get_my_position(game_state)
        self.save_my_location[((1200-game_state.data.timeleft)//4) % 15] = agent_location
        edibles_positions = self.get_edibles(game_state)
        enemy_ghosts, scared_enemy_ghosts = self.get_enemy_ghosts_agent_can_see(game_state, agent_location)

        # If the agent has enough food to secure a win or time is short , return home..
        if self.should_go_home(game_state, enemy_ghosts):
            print(f"Returning home...enemy ghosts {enemy_ghosts}")
            return self.go_home(game_state, agent_location, closest_ghost=enemy_ghosts[0] if enemy_ghosts else None)

        # If there are scared enemy ghosts, it means that we ate a power-up, so go either after food indisturbed or after the closest enemy
        if scared_enemy_ghosts and not enemy_ghosts:
            print("The enemy is scared, going after food or enemies.")
            # If there is not much food left, prioritize food, no scared enemies
            return self.handle_scared_enemies(game_state, edibles_positions, scared_enemy_ghosts)

        # If it's being chased by an enemy ghost, plan an escape strategy
        if enemy_ghosts:
            print("Enemy ghosts detected, planning escape strategy.")
            return self.plan_escape_strategy(game_state, enemy_ghosts)

        ### Not being chased ###

        # Even though you are not being chased, you might have escaped before you were going for a capsule. Check if that's the case and if it is continue to go.
        if self.keep_going_for_capsule(game_state):
            return self.Astar(game_state, self.capsule)

                    
        # Default: Collect food.
        print(f"I'm going after food...{edibles_positions[0]}")
        return self.Astar(game_state, edibles_positions[0])

    def should_go_home(self, game_state, enemy_ghosts):
        """
        Determines whether the agent should prioritize returning home.
        """
        if self.get_food_count(game_state) <= 2:
            print("Going home to secure a win.")
            return True

        carrying = self.num_carrying(game_state)
        if carrying and self.check_timeleft_wrt_distance(game_state):
            print("Time is running out, returning home.")
            return True

        return False
    
    def handle_scared_enemies(self, game_state, edibles_positions, scared_enemy_ghosts):
        if self.get_food_count(game_state) <= OffensiveReflexAgent.FOOD_CARRYING_THRESHOLD:
                print("Don't care about scared enemies...")
                for food in edibles_positions:
                    if food not in scared_enemy_ghosts:
                        return self.Astar(game_state, food)
                    
        return self.Astar(game_state, edibles_positions[0])
    
    def keep_going_for_capsule(self, game_state):

        capsules = self.get_capsules(game_state)
        if self.capsule not in capsules:
            self.capsule = None
            
        if self.capsule is not None:
            print("I was being chased, better to go after a power-up...")
            return True
        return False
    
    def defensive_action(self, game_state):
        """
        Determines the best defensive action when the agent is a ghost.
        """
        agent_location = self.get_my_position(game_state)
        self.save_my_location[((1200-game_state.data.timeleft)//4) % 15] = agent_location
        enemy_pacmen = self.get_enemy_pacmen_agent_can_see(game_state, agent_location)
            # Get only ghosts that the attacker can see
        enemy_ghosts, _ = self.get_enemy_ghosts_agent_can_see(game_state, agent_location)
        edibles_positions = self.get_edibles(game_state)

        if enemy_pacmen:
            print("Enemy Pacmen detected on my side.")
            action = self.handle_enemy_pacmen(game_state, enemy_pacmen, edibles_positions)
            if action is not None:
                return action
        
        if game_state.data.timeleft < 1200 - 4 * len(self.save_my_location):
            # If it was in a deadlock, handle the deadlock
            if self.handle_deadlock(agent_location):
                return self.Astar(game_state, self.escape_deadlock_cell)
            
            # If you see enemy ghosts and you're a ghost
            # if enemy_ghosts:
            if len(set(self.save_my_location)) < 5:
                print("Enemy ghosts detected, planning escape strategy.")
                c = None
                if enemy_ghosts:
                    c = enemy_ghosts[0]
                elif enemy_pacmen:
                    c = enemy_pacmen[0]
                else:
                    c = agent_location
                return self.get_deadlock_cell(game_state, c)



        # Go for closest food if there is no rush in escaping or killing enemy. Do your duty as an attacker!
        print(f"I'm going after food...{edibles_positions[0]}")
        return self.Astar(game_state, edibles_positions[0])
    
    def handle_enemy_pacmen(self, game_state, enemy_pacmen, edibles_positions):
        """
        Handles the presence of enemy Pacmen on the agent's side.
        """
        # If the agent is scared, go after food in the other half.
        if self.scared(game_state):
            print("I'm scared, going after food...")
            return self.Astar(game_state, edibles_positions[0])
        
        # Compute how much food enemy's attacker is carrying
        if self.enemy_is_carrying_a_lot_of_food(game_state):
            return self.Astar(game_state, enemy_pacmen[0])
    
    def enemy_is_carrying_a_lot_of_food(self, game_state):
        """
        Returns True if the enemy is carrying a lot of food.
        """
        # Get food we ate - 
        # TOTAL_FOOD = food_we_are_carrying + food_we_ate + food_left_to_eat
        total_food = self.total_food_enemies
        food_we_are_carrying = self.num_carrying(game_state) # + food our defender is carrying
        food_left_to_eat = self.get_food_count(game_state)
        food_we_ate = total_food - food_left_to_eat - food_we_are_carrying

        # Score = RED_food_eaten - BLUE_food_eaten
        score = game_state.data.score

        # Get food they are carrying 
        # TOTAL_FOOD = food_enemy_carrying + food_enemy_ate + food_we_are_defending
        food_we_are_defending = self.get_our_food(game_state)
        # If we are red, BLUE_food_eaten = RED_food_eaten - score. If we are blue, RED_food_eaten = BLUE_food_eaten + score
        food_enemy_ate = food_we_ate - score if self.red else food_we_ate + score
        food_enemy_carrying = self.total_food_enemies - len(food_we_are_defending) - food_enemy_ate
        
        # If he's carrying a lot of food, go after the enemy pacmen
        if food_enemy_carrying > OffensiveReflexAgent.FOOD_CARRYING_THRESHOLD: # How to define a lot(?)
            # Become defender
            print("Enemy is carrying a lot of food. I'm going after him...")
            return True
        return False

    def handle_deadlock(self, agent_location):
        if self.escape_deadlock:
            if agent_location == self.escape_deadlock_cell:
                print("I reached the deadlock cell.")
                self.escape_deadlock = False
            else:
                print("I'm still exiting a deadlock.")
                print("My location:", agent_location)
                print("Escape cell:", self.escape_deadlock_cell)
                return True
        return False
    
    def reachable_food(self, agent_location, closest_ghost, edibles):
        for food in edibles:
            distance_between_my_ghost_and_food = self.get_maze_distance(agent_location, food)
            distance_between_enemy_ghost_and_food = self.get_maze_distance(food, closest_ghost)

            # Go after food only if you can manage to get it and go back home before the ghost catches you
            if distance_between_my_ghost_and_food < distance_between_enemy_ghost_and_food:
                return food
        return None
    
    def handle_ghosts(self, game_state, enemy_ghosts, edibles_positions, agent_location):
        #print("I see enemy ghosts.")
        # Go after food only if you can manage to get it and go back home before the ghost catches you
        food = self.reachable_food(agent_location, enemy_ghosts[0], edibles_positions)
        if food is not None:
            return self.Astar(game_state, food)
        
        # If no food is reachable, go N cells away from the defender
        return self.get_deadlock_cell(game_state, enemy_ghosts[0])
        

    def get_deadlock_cell(self, game_state, closest_ghost):
            N = 10
            dx = -2 if self.red else 2
            new_loc = [closest_ghost[0] + dx, closest_ghost[1]]
            while True:
                escape_cells = self.get_edge_home_cells(game_state, get_idx=new_loc[0])
                #escape_cells.extend(self.get_edge_bottom_up_cells(game_state))
                escape_cells.sort(key=lambda cell: (self.get_maze_distance(closest_ghost, cell), -cell[0] if self.red else cell[0]) )
                print(escape_cells)

                for cell in escape_cells:
                    if self.get_maze_distance(cell, closest_ghost) > N:
                        self.escape_deadlock_cell = cell
                        self.escape_deadlock = True
                        print("Closest ghost:", closest_ghost)
                        print("Escape cell:", cell)
                        return self.Astar(game_state, cell, closest_ghost)
                dx = -1 if self.red else 1
                new_loc[0] += dx

    def go_home(self, game_state, agent_location=None, closest_ghost=None):
        """
        Returns the best action to take to go home.
        """

        if closest_ghost:
            print("There are ghosts!")
            closest_home = self.get_closest_reachable_home_cell_position(game_state, agent_location, closest_ghost)
            if not closest_home:
                # Lure the defender away
                print("I'm luring the defender away...")
                return self.Astar(game_state, self.get_farthest_home_cell_position(game_state))
        else:
            closest_home = self.get_closest_home_cell_position(game_state)

        if len(set(self.save_my_location)) < 5:
            middle_point = (closest_home[0] + agent_location[0]) // 2, (closest_home[1] + agent_location[1]) // 2
            nearest_point = nearest_open_space(game_state, middle_point)
            return self.Astar(game_state, nearest_point)

        return self.Astar(game_state, closest_home)
    
    def plan_escape_strategy(self, game_state, closest_ghosts):
        """
        Plans an escape strategy when being chased by enemy ghosts.
        """
        # If a capsule can be reached before the ghost catches me, go for it
        agent_location = self.get_my_position(game_state)
        action_to_capsule = self.go_to_capsule(game_state, agent_location, closest_ghosts)
        if action_to_capsule:
            print("I'm going after a capsule to escape...")
            return action_to_capsule

        # Retrieve locations of closest ghost and closest reachable home cell
        closest_ghost = closest_ghosts[0]
        closest_home = self.get_closest_reachable_home_cell_position(game_state, agent_location, closest_ghost)
        if not closest_home:
            # Lure the defender away
            farthest_home =  self.get_farthest_home_cell_position(game_state)
            print("No safe home cell. I'm luring the defender away...{farthest_home}")
            #return self.Astar(game_state, farthest_home)
            closest_home = farthest_home

        # Go after food only if you can manage to get it and go back home before the ghost catches you
        action_to_food = self.go_to_food(game_state, agent_location, closest_ghost, closest_home)
        if action_to_food:
            print("I'm going after food because I know I can escape escape...")
            return action_to_food
        
        if self.handle_deadlock(agent_location):
                return self.Astar(game_state, self.escape_deadlock_cell)
            
        # If you see enemy ghosts and you're a ghost
        # if enemy_ghosts:
        if len(set(self.save_my_location)) < 5:
            print("Enemy ghosts detected, planning escape strategy.")
            # if distance between ghost and me is more than 2, get closer.
            if self.get_maze_distance(agent_location, closest_ghost) > 2:
                return self.Astar(game_state, closest_ghost)
            else:
                return self.get_deadlock_cell(game_state, closest_ghost)


        
        # If no food is reachable, go back home
        print("No food is reachable. I'm going home...")
        return self.Astar(game_state, closest_home)
    
    def go_to_capsule(self, game_state, agent_location, enemy_ghosts):
        capsules = self.get_capsules(game_state)
        for capsule in capsules:
            distance_to_capsule = self.get_maze_distance(agent_location, capsule)
            print(f"My distance {agent_location} to capsule {capsule}: {distance_to_capsule}")
            ghost_to_capsule = min(self.get_maze_distance(capsule, ghost) for ghost in enemy_ghosts)
            print(f"Ghost {enemy_ghosts} distance to capsule: {ghost_to_capsule}")
            if distance_to_capsule < ghost_to_capsule:
                self.capsule = capsule
                return self.Astar(game_state, capsule)
        return None
    
    def go_to_food(self, game_state, agent_location, closest_ghost, closest_home):
        # distance_between_ghost_and_my_home_cell = self.get_maze_distance(closest_ghost, closest_home)
        edibles_positions = self.get_edibles(game_state)    # food or enemy ghosts if not scared
        for food in edibles_positions:
            distance_between_pacman_and_food = self.get_maze_distance(agent_location, food)
            distance_between_food_and_my_home_cell = self.get_maze_distance(food, closest_home)

            open_space = nearest_open_space(game_state, food)
            distance_between_ghost_and_open_space = self.get_maze_distance(closest_ghost, open_space)
            distance_between_food_and_open_space = self.get_maze_distance(food, open_space)

            # Go after food only if you can manage to get it and go back home before the ghost catches you
            if distance_between_pacman_and_food + distance_between_food_and_open_space < distance_between_ghost_and_open_space:#distance_between_ghost_and_my_home_cell:
                return self.Astar(game_state, food)
            
            # The edibles list is ordered by distance, so if the food is too far away, break
            #if distance_between_pacman_and_food > distance_between_ghost_and_my_home_cell:
                #break

            #if distance_between_food_and_my_home_cell > distance_between_ghost_and_my_home_cell:
             #   break
        return None
    
    def check_timeleft_wrt_distance(self, game_state):
        """
        Checks if the timeleft is barely enough to go back home.
        """
        agent_location = self.get_my_position(game_state)
        closest_home = self.get_closest_home_cell_position(game_state)
        distance_to_home = self.get_maze_distance(agent_location, closest_home)
        return distance_to_home * OffensiveReflexAgent.TIME_MARGIN_FACTOR  < game_state.data.timeleft < distance_to_home * OffensiveReflexAgent.TIME_MARGIN_FACTOR*1.5    # 1.5 times distance to home



class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    Cool Defender
    """
    def choose_action(self, game_state):
        # start = time.time()
        print("-" * 50)
        print("Defender")

        # If the game just started, predict the most likely target for enemy attacker and go there
        if game_state.data.timeleft > 1050: # can be made better
            closest_food_to_enemy = self.predict_initial_target(game_state)
            return closest_food_to_enemy
        
        # If the agent is a ghost, meaning that is on its own side
        excluded_cells = self.get_edge_home_cells(game_state, enemy_home=True)
        if not self.is_pacman(game_state):
            # Plan defense

            # if you are scared, go in attack after the closest food
            if self.scared(game_state):
                print("I'm scared, going after food...")
                closest_food = self.get_edibles(game_state)
                return self.Astar(game_state, closest_food[0])
            
            # If you see any pacman, follow him
            enemy_pacmen = self.get_enemy_pacmen_positions(game_state)
            # print(self.save_my_location)
            if enemy_pacmen:
                print(f"Chasing enemy pacman...{enemy_pacmen[0]}")
                #my_loc = self.get_my_position(game_state)
                #if my_loc not in self.save_my_location:
                #    self.save_my_location.add(my_loc)
                #else:
                #    print(f"I'm stuck in a loop. Don't pass from this cell anymore...{my_loc}")
                    # excluded_cells.extend(list(self.save_my_location))
                # closest_food_to_enemy = self.go_to_closest_food_from_enemy_position(self, game_state, enemy_pacmen[0])
                # print(excluded_cells)
                return self.Astar(game_state, enemy_pacmen[0], excluded_cells)
            
            #self.save_my_location.clear()

        else:
            # The defender agent is a pacman
            print("I'm a pacman.")
            print("My location:", self.get_my_position(game_state))
            if self.get_food_count(game_state) <= 2:
                print("Going home to secure a win.")
                closest_home = self.get_closest_home_cell_position(game_state)
                return self.Astar(game_state, closest_home)
            

            # If the enemy is eating food, go after him by predicting the next food he's going to eat and going there
            predicted_enemy_location = self.predict_enemy_from_food_disappearance(game_state)
        
            closest_food = self.get_edibles(game_state)
            print(self.scared(game_state), self.enemy_is_attacking)
            if self.scared(game_state) or not self.enemy_is_attacking:
                print("I'm scared or the enemy is not attacking...")
                return self.Astar(game_state, closest_food[0])
            
            if predicted_enemy_location:
                print(f"Predicting enemy location...{predicted_enemy_location}")
                closest_food_to_enemy = self.closest_food_to_location(game_state, predicted_enemy_location)
                print("Closest food to enemy:", closest_food_to_enemy)
                action = self.Astar(game_state, closest_food_to_enemy)
                print("Action:", action)
                return action
            
            # If the enemy is not attacking, keep going after the closest food
            #print("Well, I'm in attack and they are not attacking, soo I'm going after food.")
            #closest_foods = self.get_edibles(game_state)
            #return self.Astar(game_state, closest_foods[0])

        # If the enemy is eating food, go after him by predicting the next food he's going to eat and going there
        predicted_enemy_location = self.predict_enemy_from_food_disappearance(game_state)
        if predicted_enemy_location:
            print(f"Predicting enemy location...{predicted_enemy_location}")
            closest_food_to_enemy = self.closest_food_to_location(game_state, predicted_enemy_location)
            return self.Astar(game_state, closest_food_to_enemy, excluded_cells)
        
        print("I'm a ghost. I'm not scared. I don't see any enemy pacman nor they are attacking.")
        print("I'm going to predict the next move of the opponent. I'm going to defend the part of my map with the highest density of food.")
        home_cells = self.get_edge_home_cells(game_state)
        food_we_are_defending = self.get_our_food(game_state)
        # find center of the most dense food cluster
        food_cluster_center = min(food_we_are_defending, key=lambda food: sum(self.get_maze_distance(food, other_food) for other_food in food_we_are_defending))
        print("Food cluster center:", food_cluster_center)
        # sort home cell by closest to the center of the most dense food cluster
        closest_entry_cell = min(home_cells, key=lambda food: self.get_maze_distance(food, food_cluster_center))

        # get random cell between the closest entry cell and the center of the most dense food cluster which is not a wall in a 4x4 window
        middle_cell = (closest_entry_cell[0] + food_cluster_center[0]) // 2, (closest_entry_cell[1] + food_cluster_center[1]) // 2

        #print(middle_cell, food_cluster_center)
        middle_cell = nearest_open_space(game_state, middle_cell)
        food_cluster_center = nearest_open_space(game_state, food_cluster_center)
        #print(closest_entry_cell, middle_cell, food_cluster_center)

        best_defense_location = random.choice([closest_entry_cell, middle_cell])
        #print(best_defense_location)

        # Update the score
        self.previous_score = game_state.data.score
        return self.Astar(game_state, best_defense_location, excluded_cells)
        


    def predict_initial_target(self, game_state):
        """
        Return the action that bring the agent to the closest food to the enemy start location
        """

        # Get enemy start location
        w, h = arena_width(game_state), arena_height(game_state)
        x_my_start_location, y_my_start_location = self.start
        enemy_start_location = w - x_my_start_location - 1, h - y_my_start_location - 1

        # Get the closest food to the enemy start location
        return self.go_to_closest_food_from_enemy_position(game_state, enemy_start_location)
    

    def go_to_closest_food_from_enemy_position(self, game_state, enemy_position):
        excluded_cells = self.get_edge_home_cells(game_state, enemy_home=True)
    
        food_positions = self.get_closest_foods_from_enemy_position(game_state, enemy_position)

        return self.Astar(game_state, food_positions[0], excluded_cells)
    

    def predict_enemy_from_food_disappearance(self, game_state):
        """
        Return True if the enemy is eating food
        """
        previous_food_we_are_defending = self.our_food
        food_we_are_defending = set(self.get_our_food(game_state))
        print("Food we are defending", food_we_are_defending)
        print("Previous food we are defending", previous_food_we_are_defending)
        food_eaten = previous_food_we_are_defending - food_we_are_defending
        if food_eaten:
            food_eaten = food_eaten.pop() if food_eaten else None
            self.our_food = food_we_are_defending
            self.last_food_eaten_by_enemy = food_eaten
            self.enemy_is_attacking = True
            self.steps_since_last_food_eaten = 0
            print("Food eaten", self.enemy_is_attacking, game_state.data.timeleft)
            return food_eaten
        
        # if we are red and the score goes down, it means that the enemy scored
        if (self.red and game_state.data.score < self.previous_score) or (not self.red and game_state.data.score > self.previous_score):
            print("Enemy scored")
            # Blue or red scored
            self.enemy_is_attacking = False
            return self.last_food_eaten_by_enemy
        
        if self.enemy_is_attacking:
            print("Enemy is attacking")
            return self.last_food_eaten_by_enemy
        
        if self.steps_since_last_food_eaten > 10:
            return None
        

        self.steps_since_last_food_eaten += 1
        return None
    
    def closest_food_to_location(self, game_state, enemy_location):
        """
        Returns the closest food to the enemy location
        """
        food_positions = self.get_our_food(game_state)
        closest_food = min(food_positions, key=lambda food: self.get_maze_distance(food, enemy_location))
        return closest_food
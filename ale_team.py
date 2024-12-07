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
        self.escape_deadlock_state = 0
        self.deadlock_cell = [0, 0]
        self.total_food_enemies = 0

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)  # Start position of the agent (int x, int y)
        CaptureAgent.register_initial_state(self, game_state)
        self.total_food_enemies = self.get_total_food_enemies(game_state)
        
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
        
    def in_lead(self, game_state):
        """
        Checks if the agent's team is leading in score. 
        Red team scores are positive, while Blue team scores are negative.
        The score of the game is the sum of the two team's scores.
        Therefore, the agent is in the lead if the score is positive.
        """
        return game_state.data.score if self.red else -game_state.data.score
    
    def near_center(self, game_state, my_x):
        """
        Checks if the agent is near the center of the arena.
        """
        w = arena_width(game_state)
        if self.red:
            x_left, x_right = w // 2 - 5, w // 2 - 1
        else:
            x_left, x_right = w // 2, w // 2 + 4
        return x_left <= my_x <= x_right

    
    ### Position and Distance Calculations ###

    def get_my_position(self, game_state):
        """
        Returns the agent's current position.
        """
        return game_state.get_agent_state(self.index).get_position()
    
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
            list: a sorted list of edibles positions [closest to the agent ... farthest].
        """
        edibles = self.get_food_positions(game_state)

        # Add enemy Pacmen positions if the agent isn't scared
        if not self.scared(game_state):
            edibles.extend(self.get_enemy_pacmen_positions(game_state))

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
        return [pos for a in enemy_states if (pos := a.get_position()) is not None and (filter_fn is None or filter_fn(a))]

    def get_enemy_pacmen_positions(self, game_state):
        """
        Returns the positions of enemy Pacmen visible to the agent. 
        This means that they are on the agent side and can collect food in its territory.
        """
        return self.get_enemy_positions(game_state, filter_fn=lambda a: a.is_pacman)

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

    def scared(self, game_state):
        """
        Returns True if the agent is currently scared.
            - Only agents in ghost mode can be scared.
            - Pacmen cannot be scared because they are in offensive mode when on the opponent's side, and power pellets do not affect them.
        """
        return game_state.get_agent_state(self.index).scared_timer != 0

    def panic(self, game_state):
        """
        Checks if the agent is in a starting position on the opponent's side.
        """
        my_pos = self.get_my_position(game_state)
        enemy_start = arena_width(game_state) - 2 if self.red else 1
        return my_pos[0] == enemy_start

    ### Strategic Helpers ###
    
    def get_edge_home_cells(self, game_state, enemy_home=False):
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
        return [(edge_col, row) for row in range(1, arena_height(game_state) - 1) if not is_wall(game_state, (edge_col, row))]

    
    def get_closest_home_cell_position(self, game_state):
        """
        Finds the closest edge cell on the agent's home side for safe return.
        
        Returns:
            Tuple: The closest escape position (x, y).
        """
        home_list = self.get_edge_home_cells(game_state)
        home = self.sort_positions_by_distance(game_state, home_list)
        return home[:1] # home[0] why not this?
    

    def get_escape_position(self, game_state):
        """
        Returns the farthest escape position when returning home. (?why farthest?)
        """
        edge_home_cells = self.get_edge_home_cells(game_state)
        edge_home_cells_distances = self.sort_positions_by_distance(game_state, edge_home_cells)
        return edge_home_cells_distances[-1]

    ### Pathfinding and Scoring ###

    def score(self, my_pos, target, distance_traveled):
        """
        Calculates a heuristic score for A* pathfinding.
        """
        f_n = distance_traveled + self.get_maze_distance(my_pos, target)   # f(n) = g(n) + h(n)
        return f_n

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
        score = self.score(self.get_my_position(game_state), target, distance_traveled)
        pq.put((score, next(counter), next_state))

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
        

class OffensiveReflexAgent2(ReflexCaptureAgent):
    """
    Cool Attacker
    """
    
    def choose_action(self, game_state):
        # start = time.time()

        # If you end up on the far side of the arena, give up
        if self.panic(game_state): return "Stop"

        targets, exclude = self.choose_targets(game_state), []
        if isinstance(targets, tuple): targets, exclude = targets

        best_action = self.choose_best_action_for_target(game_state, targets, exclude)
        
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        return best_action
    

    def choose_targets(self, game_state):
        # First thing first, check if you should go home
        home_cell = self.go_home(game_state)
        if home_cell:
            return home_cell

        # Get relevant game state information.
        carrying = self.num_carrying(game_state)
        edibles_positions = self.get_edibles(game_state)
        closest_home = self.get_closest_home_cell_position(game_state)
        capsules = self.get_capsules(game_state)
        enemy_pacmen = self.get_enemy_pacmen_positions(game_state)
        ghosts, scared_ghosts = self.get_enemy_ghost_positions(game_state)
        
        # Handle deadlock state.
        if self.handle_deadlock(game_state, enemy_pacmen):
            return enemy_pacmen
        
        # If we are escaping, go to deadlock cell (stop escaping when we reach that cell)
        if self.get_my_position(game_state) == self.deadlock_cell:
            self.escape_deadlock_state = 0
        
        # If your already on your side and are running, if there are enemies you can chase, go after them
        if self.escape_deadlock_state and enemy_pacmen and not self.scared(game_state):
            self.escape_deadlock_state = 2
            return enemy_pacmen
        
        if self.escape_deadlock_state == 1:
            # We wish to exclude enemy home edge cells when going to escape position
            if self.in_lead(game_state) > 0:
                return [self.deadlock_cell], self.get_edge_home_cells(game_state, enemy_home=True)  
            else:
                return edibles_positions
        else:
            # No more deadlock
            self.escape_deadlock_state = 0

        # If you dont see anyone, go get food
        if not scared_ghosts and not ghosts:
            return edibles_positions

        # If you see enemy ghosts and there is a capsule, go get the capsule
        if ghosts and capsules:
            return capsules

        # If you see a ghost and there is no capsules
        if ghosts:
            # If a ghost is near you and you have food, play safe and start going home (otherwise take the risk)
            if carrying and min(self.get_distances_to(game_state, ghosts)) < 3:
                self.returning_home = True
                return closest_home
            
        # If none of the above conditions happened then go for food
        return edibles_positions

    def go_home(self, game_state):
        """
        Determines if the agent should return home based on game conditions.
        """
        carrying = self.num_carrying(game_state)
        closest_home = self.get_closest_home_cell_position(game_state)

        # If the agent has enough food to secure a win, return home.
        if self.get_food_count(game_state) <= 2:
            return closest_home

        # If time is running out and the agent is carrying food, return home.
        if carrying and game_state.data.timeleft < 100:
            return closest_home

        # If the agent has dropped off food, stop returning home.
        if not carrying:
            self.returning_home = False

        # Continue returning home if already decided.
        if self.returning_home:
            return closest_home

        return False
    
    def handle_deadlock(self, game_state, enemy_pacmen):
        """
        Handles the agent's behavior in a deadlock situation.
        """
        # If the agent is in its deadlock cell, reset the escape state.
        if self.get_my_position(game_state) == self.deadlock_cell:
            self.escape_deadlock_state = 0

        # If in an escape state and enemies are nearby, prioritize chasing them.
        if self.escape_deadlock_state and enemy_pacmen and not self.scared(game_state):
            self.escape_deadlock_state = 2
            return True

        # If escape state 1, plan to escape or focus on food collection.
        if self.escape_deadlock_state == 1:
            if self.in_lead(game_state) > 0:
                return [self.deadlock_cell], self.get_edge_home_cells(game_state, enemy_home=True)
            return False

        # Reset deadlock state if none of the conditions apply.
        self.escape_deadlock_state = 0
        return False
    

    def choose_best_action_for_target(self, game_state, targets, excluded_positions):
        ghosts, scared_ghosts = self.get_enemy_ghost_positions(game_state)
        my_pos = self.get_my_position(game_state)
        ghost_dists = self.get_distances_to(game_state, ghosts)

        if len(targets) == 1 or not ghost_dists:
            return self.Astar(game_state, targets[0], excluded_positions)
        else:   # note that ghost dists is not empty!
            for target in targets:
                open_space = nearest_open_space(game_state, target)
                ghost_dist = min(ghost_dists)

                my_dist = self.get_maze_distance(my_pos, open_space)
                margin = self.get_maze_distance(open_space, target)
                
                if my_dist + margin * 2 + 1 < ghost_dist:
                    return self.Astar(game_state, target, excluded_positions)

            if not self.is_pacman(game_state) and min(ghost_dists) <= 5:
                self.escape_deadlock_state = 1
                self.deadlock_cell = self.get_escape_position(game_state) 
                return self.Astar(game_state, self.deadlock_cell, self.get_edge_home_cells(game_state, enemy_home=True))    
            else:
                return self.Astar(game_state, targets[-1], excluded_positions)


class OffensiveReflexAgent(ReflexCaptureAgent):
    def choose_action(self, game_state):
        # If the agent is pacman, meaning that is the opponent's side
        if self.is_pacman(game_state):

            # If the agent has enough food to secure a win, return home.
            if self.get_food_count(game_state) <= 2:
                return self.go_home(game_state)
            
            enemy_ghosts, scared_enemy_ghosts = self.get_enemy_ghost_positions(game_state)
            edibles_positions = self.get_edibles(game_state)

            # If there are scared enemy ghosts, it means that we ate a power-up, so go either after food indisturbed or after the closest enemy
            if scared_enemy_ghosts and not enemy_ghosts:
                return self.Astar(game_state, edibles_positions[0])
            
            # If it's being chased by an enemy ghost, plan an escape strategy
            if enemy_ghosts:
                best_action = self.plan_escape_strategy(game_state, enemy_ghosts)
                return best_action

            ### Not being chased ###
            agent_location = self.get_my_position(game_state)
            closest_home = self.get_closest_home_cell_position(game_state)
            distance_to_home = self.get_maze_distance(agent_location, closest_home)
            if game_state.data.timeleft < distance_to_home * 6:    # 1.5 times distance to home
                return self.go_home(game_state)
            
            return self.Astar(game_state, edibles_positions[0])
            
        else:
            # The agent is a ghost
            edibles_positions = self.get_edibles(game_state)

            # Check if there are enemy pacmen on the agent's side (also the defender should communicate for pacman checking)
            enemy_pacmen = self.get_enemy_pacmen_positions(game_state)
            if enemy_pacmen:
                # If the agent is scared, go after food in the other half.
                if self.scared(game_state):
                    agent_location = self.get_my_position(game_state)
                    for food in edibles_positions:
                        distance_between_me_and_food = self.get_maze_distance(agent_location, food)
                        distance_between_enemy_pacman_and_food = self.get_maze_distance(enemy_pacmen, food)
                        if distance_between_me_and_food < distance_between_enemy_pacman_and_food:
                            return self.Astar(game_state, food)
                        
                # Communicate with defender, see if he needs help
                # Compute how much food the attacker's carrying
                my_food = self.get_our_food(game_state)
                enemy_carrying = self.total_food_enemies - len(my_food) - self.in_lead(game_state)
                
                # If he's carrying a lot of food, go after the enemy pacmen

            # Go for closest food if there is no rush in escaping or killing enemy. Do your duty as an attacker!
            return self.Astar(game_state, edibles_positions[0])
    
    def go_home(self, game_state):
        """
        Returns the best action to take to go home.
        """
        closest_home = self.get_closest_home_cell_position(game_state)

        return self.Astar(game_state, closest_home[0])
    
    def plan_escape_strategy(self, game_state, enemy_ghosts):
        """
        Plans an escape strategy when being chased by enemy ghosts.
        """
        # If I can reach a capsule before the ghost catches me, go for it
        agent_location = self.get_my_position(game_state)
        capsules = self.get_capsules(game_state)
        for capsule in capsules:
            distance_to_capsule = self.get_maze_distance(agent_location, capsule)
            ghost_to_capsule = min(self.get_maze_distance(capsule, ghost) for ghost in enemy_ghosts)
            if distance_to_capsule < ghost_to_capsule:
                return self.Astar(game_state, capsule)

        # Carefully plan the next step.
        closest_home = self.get_closest_home_cell_position(game_state)
        distance_between_ghost_and_my_home_cell = min(self.get_maze_distance(closest_home, ghost) for ghost in enemy_ghosts)
        edibles_positions = self.get_edibles(game_state)    # food or enemy ghosts if not scared
        for food in edibles_positions:
            distance_between_pacman_and_food = self.get_maze_distance(agent_location, food)
            distance_between_food_and_my_home_cell = self.get_maze_distance(food, closest_home)

            # Go after food only if you can manage to get it and go back home before the ghost catches you
            if distance_between_pacman_and_food + distance_between_food_and_my_home_cell < distance_between_ghost_and_my_home_cell:
                return self.Astar(game_state, food)
            
            # The edibles list is ordered by distance, so if the food is too far away, break
            if distance_between_pacman_and_food > distance_between_ghost_and_my_home_cell:
                break

            if distance_between_food_and_my_home_cell > distance_between_ghost_and_my_home_cell:
                break
        
        # If no food is reachable, go back home
        return self.Astar(game_state, closest_home)
    
    def get_closest_home_cell_position(self, game_state):
        return super().get_closest_home_cell_position(game_state)[0]

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    Cool Defender
    """
    def choose_action(self, game_state):
        # start = time.time()

        targets, exclude = self.choose_targets(game_state), []
        if isinstance(targets, tuple): targets, exclude = targets

        best_action = self.choose_best_action_for_target(game_state, targets, exclude)
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        return best_action
    
    def choose_targets(self, game_state):

        h = self.get_closest_home_cell_position(game_state)
        p = self.get_enemy_pacmen_positions(game_state)
        
        ghosts, scared_ghosts = self.get_enemy_ghost_positions(game_state)
        home_cells = self.get_edge_home_cells(game_state)
        excluded_cells = self.get_edge_home_cells(game_state, enemy_home=True)
        
        if game_state.data.timeleft > 1100:
            return [h[len(h) // 2 - 1]], excluded_cells

        if p and self.near_center(game_state, self.get_my_position(game_state)[0]):
            return self.sort_positions_by_distance(game_state, p)[:1], excluded_cells

        if p:
            closest_cell = self.get_ghost_facing_home_cell(game_state, p)
            return [closest_cell], excluded_cells
        
        if ghosts or scared_ghosts:
            closest_cell = self.get_ghost_facing_home_cell(game_state, ghosts + scared_ghosts)
            return [closest_cell], excluded_cells

        return [random.choice(home_cells)], excluded_cells

    def get_ghost_facing_home_cell(self, game_state, ghosts):
        enemy_pos = self.sort_positions_by_distance(game_state, ghosts)[0]
        w = arena_width(game_state)
        possible_x = range(w // 2 - 1, -1, -1) if self.red else range(w // 2, w)

        for x in possible_x:
            curr_cell = (x, enemy_pos[1])
            if not is_wall(game_state, curr_cell):
                return curr_cell

    def choose_best_action_for_target(self, game_state, targets, excluded_positions):
        return self.Astar(game_state, targets[0], excluded_positions)
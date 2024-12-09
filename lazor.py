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
    
def nearest_open_space(game_state, food):
    stack = Stack()
    stack.push(food)
    visited = set()
    visited.add(food)

    while not stack.is_empty():
        curr_cell = stack.pop()
        if not surrounded_by_walls(game_state, curr_cell):
            return curr_cell
        
        for dx, dy in zip([-1, 1, 0, 0], [0, 0, -1, 1]):
            new_cell = tuple([curr_cell[0] + dx, curr_cell[1] + dy])
            if is_not_wall(game_state, new_cell) and new_cell not in visited:
                stack.push(new_cell)
                visited.add(new_cell)
    
def surrounded_by_walls(game_state, cell):
    wall_cnt = 0
    wall_map = game_state.get_walls()
    for dx, dy in zip([-1, 1, 0, 0], [0, 0, -1, 1]):
        if wall_map[int(cell[0]) + dx][int(cell[1]) + dy]:
            wall_cnt += 1
    return wall_cnt > 1

def is_wall(game_state, cell):
    wall_map = game_state.get_walls()
    return wall_map[int(cell[0])][int(cell[1])]

def is_not_wall(game_state, cell):
    wall_map = game_state.get_walls()
    return not wall_map[int(cell[0])][int(cell[1])]

def arena_width(game_state):
    return game_state.get_walls().width

def arena_height(game_state):
    return game_state.get_walls().height

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

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        
    def get_next_game_state(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        next_game_state = game_state.generate_successor(self.index, action)
        pos = next_game_state.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return next_game_state.generate_successor(self.index, action)
        else:
            return next_game_state
        
    def in_lead(self, game_state):
        return game_state.data.score if self.red else -game_state.data.score
        
    def num_carrying(self, game_state):
        return game_state.get_agent_state(self.index).num_carrying
    
    def not_scared(self, game_state):
        return game_state.get_agent_state(self.index).scared_timer == 0
    
    def scared(self, game_state):
        return game_state.get_agent_state(self.index).scared_timer != 0
    
    def get_my_position(self, game_state):
        return game_state.get_agent_state(self.index).get_position()
    
    def get_food_positions(self, game_state):
        return self.get_food(game_state).as_list()
    
    def get_food_count(self, game_state):
        return len(self.get_food(game_state).as_list())
    
    def get_enemy_pacmen_positions(self, game_state):
        enemy_states = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        return [a.get_position() for a in enemy_states if a.is_pacman and a.get_position() is not None]
    
    def get_enemy_ghost_positions(self, game_state):
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        scared_ghosts = [a.get_position() for a in enemies if not a.is_pacman and a.scared_timer > 5 and a.get_position() is not None]
        ghosts = [tuple(map(int, a.get_position())) for a in enemies if not a.is_pacman and a.scared_timer <= 5 and a.get_position() is not None]
        return ghosts, scared_ghosts
    
    def calculate_distances_to(self, game_state, positions):
        return [self.get_maze_distance(self.get_my_position(game_state), pos) for pos in positions]
    
    def calculate_distances_with_positions(self, game_state, positions):
        return [(self.get_maze_distance(self.get_my_position(game_state), pos), pos) for pos in positions]
    
    def sort_distances(self, distances):
        return [dist[1] for dist in sorted(distances)]
    
    def is_pacman(self, game_state):
        state = game_state.get_agent_state(self.index)
        return state.is_pacman

    def get_edibles(self, game_state):
        food_list = self.get_food_positions(game_state)

        # If you are closer to an enemy that you can eat than to any food, eat them
        # pacmen_list = self.get_enemy_pacmen_positions(game_state)
        pacmen_list = []
        # If you are scared, ignore attacking enemies
        if self.not_scared(game_state):
            food_list += pacmen_list

        food = self.sort_distances(self.calculate_distances_with_positions(game_state, food_list))

        return food
    
    def get_edge_home_cells(self, game_state, enemy_home=False):
        if enemy_home:
            w = arena_width(game_state) // 2 - (0 if self.red else 1)
        else:
            w = arena_width(game_state) // 2 - (1 if self.red else 0)
        home = []
        for h in range(1, arena_height(game_state) - 1):
            if is_not_wall(game_state, (w, h)):
                home.append((w, h))
        return home

    def get_escape_position(self, game_state):
        my_pos = self.get_my_position(game_state)
        edge_home_cells = self.get_edge_home_cells(game_state)
        edge_home_cells_distances = self.calculate_distances_with_positions(game_state, edge_home_cells)
        edge_home_cells_distances = self.sort_distances(edge_home_cells_distances)
        return edge_home_cells_distances[-1]

    def get_closest_home_cell_position(self, game_state):
        home_list = self.get_edge_home_cells(game_state)
        distances_w_position_list = self.calculate_distances_with_positions(game_state, home_list)
        home = self.sort_distances(distances_w_position_list)
        return home[:1]
    
    def near_center(self, game_state, my_x):
        w = arena_width(game_state)
        if self.red:
            x_left, x_right = w // 2 - 5, w // 2 - 1
        else:
            x_left, x_right = w // 2, w // 2 + 4 
        return x_left <= my_x <= x_right
        
    def panic(self, game_state):
        my_pos = self.get_my_position(game_state)
        enemy_start = arena_width(game_state) - 2 if self.red else 1
        return my_pos[0] == enemy_start
    
    def score(self, my_pos, target, distance_traveled):
        # print(my_pos, target, distance_traveled)
        my_score = self.get_maze_distance(my_pos, target) + distance_traveled
        return my_score

    def Astar(self, game_state, target, excluded_positions=None):
        pq = PriorityQueue()
        # Needed to resolve order in the priority queue if two states have the same score
        counter = count()
        self.add_to_prio_queue(pq, game_state, 0, None, None, target, counter)
        
        visited = set()
        # Positions that we don't want to visit are treated as visited
        visited.add(self.start)
        if excluded_positions:
            visited.update(excluded_positions)
        if self.get_my_position(game_state) == target:
            return "Stop"

        while not pq.empty():
            previous, current_game_state, distance_traveled = self.get_from_prio_queue(pq)
            my_pos = self.get_my_position(current_game_state)

            if my_pos == target:
                break

            actions = current_game_state.get_legal_actions(self.index)
            for action in actions:
                next_game_state = self.get_next_game_state(current_game_state, action)
                my_pos = self.get_my_position(next_game_state)
                if my_pos in visited:
                    continue
                visited.add(my_pos)
                self.add_to_prio_queue(pq, next_game_state, distance_traveled + 1, previous, action, target, counter)

        return self.first_action(previous)
    
    def add_to_prio_queue(self, pq, game_state, distance_traveled, previous, action, target, counter):
        next_state = {
            "game_state" : game_state,
            "distance_traveled" : distance_traveled,
            "previous" : previous,
            "action" : action
        }
        pq.put((self.score(self.get_my_position(game_state), target, 0), next(counter), next_state))

    def get_from_prio_queue(self, pq):
        score, order, previous = pq.get()
        current_game_state, distance_traveled = previous["game_state"], previous["distance_traveled"]
        return previous, current_game_state, distance_traveled
    
    def first_action(self, previous):
        while previous["previous"]["previous"]:
            previous = previous["previous"]
        return previous["action"]
        
class OffensiveReflexAgent(ReflexCaptureAgent):
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
        food_left = self.get_food_count(game_state)
        carrying = self.num_carrying(game_state)

        f = self.get_edibles(game_state)
        h = self.get_closest_home_cell_position(game_state)
        c = self.get_capsules(game_state)
        p = self.get_enemy_pacmen_positions(game_state)
        
        ghosts, scared_ghosts = self.get_enemy_ghost_positions(game_state)
        
        # If you have enough food to win, go home
        if food_left <= 2:
            return h

        # If we carry food and time is running out then it's better to go home 
        if carrying and game_state.data.timeleft < 100:
            return h

        # If you decided to play safe and go home, go home (if you returned all of the food you were carrying stop going home)
        if not carrying:
            self.returning_home = False

        if self.returning_home:
            return h
        
        # If we are escaping, go to deadlock cell (stop escaping when we reach that cell)
        if self.get_my_position(game_state) == self.deadlock_cell:
            self.escape_deadlock_state = 0
        
        # If your already on your side and are running, if there are enemies you can chase, go after them
        if self.escape_deadlock_state and p and self.not_scared(game_state):
            self.escape_deadlock_state = 2
            return p
        
        if self.escape_deadlock_state == 1:
            # We wish to exclude enemy home edge cells when going to escape position
            if self.in_lead(game_state) > 0:
                return [self.deadlock_cell], self.get_edge_home_cells(game_state, enemy_home=True)  
            else:
                return f
        else:
            # No more deadlock
            self.escape_deadlock_state = 0

        # If you dont see anyone, go get food
        if not scared_ghosts and not ghosts:
            return f

        # If you see enemy ghosts and there is a capsule, go get the capsule
        if ghosts and c:
            return c

        # If you see a ghost and there is no capsules
        if ghosts:
            # If a ghost is near you and you have food, play safe and start going home (otherwise take the risk)
            if carrying and min(self.calculate_distances_to(game_state, ghosts)) < 3:
                self.returning_home = True
                return h
            
        # If none of the above conditions happened then go for food
        return f
    
    def choose_best_action_for_target(self, game_state, targets, excluded_positions):
        ghosts, scared_ghosts = self.get_enemy_ghost_positions(game_state)
        my_pos = self.get_my_position(game_state)
        ghost_dists = self.calculate_distances_to(game_state, ghosts)

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
            return self.sort_distances(self.calculate_distances_with_positions(game_state, p))[:1], excluded_cells

        if p:
            closest_cell = self.get_ghost_facing_home_cell(game_state, p)
            return [closest_cell], excluded_cells
        
        if ghosts or scared_ghosts:
            closest_cell = self.get_ghost_facing_home_cell(game_state, ghosts + scared_ghosts)
            return [closest_cell], excluded_cells

        return [random.choice(home_cells)], excluded_cells

    def get_ghost_facing_home_cell(self, game_state, ghosts):
        enemy_pos = self.sort_distances(self.calculate_distances_with_positions(game_state, ghosts))[0]
        w = arena_width(game_state)
        possible_x = range(w // 2 - 1, -1, -1) if self.red else range(w // 2, w)

        for x in possible_x:
            curr_cell = (x, enemy_pos[1])
            if is_not_wall(game_state, curr_cell):
                return curr_cell

    def choose_best_action_for_target(self, game_state, targets, excluded_positions):
        return self.Astar(game_state, targets[0], excluded_positions)
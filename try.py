# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import sys
sys.path.append('./pacman-contest/src/contest')

import util as util
from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point
from game import Grid
from distance_calculator import Distancer


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
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########
class SmartCaptureAgent(CaptureAgent):
    """
    An enhanced CaptureAgent with utilities for dynamic decision-making,
    escape planning, and coordination.
    """

    def __init__(self, index, time_for_computing=.1):
        """
        Initialize the agent with additional utilities and shared state.
        """
        super().__init__(index, time_for_computing)
        self.shared_data = {}  # Shared state for communication between agents

    def register_initial_state(self, game_state):
        """
        Called once at the beginning of the game to initialize the agent.
        """
        super().register_initial_state(game_state)
        # self.distancer holds Precomputed maze distances
        self.start_position = game_state.get_agent_position(self.index)
        self.team_indices = self.get_team(game_state)
        self.opponent_indices = self.get_opponents(game_state)

    ########################
    # Opponent Tracking    #
    ########################
    def get_opponent_positions(self, game_state):
        """
        Returns estimated positions of opponents using noisy observations.
        """
        positions = []
        for idx in self.opponent_indices:
            pos = game_state.get_agent_position(idx)
            if pos:  # Opponent is visible
                positions.append(pos)
            else:  # Use noisy distance to approximate location
                noisy_distances = game_state.get_agent_distances()
                if noisy_distances and noisy_distances[idx] is not None:
                    possible_positions = self.get_possible_positions(
                        self.index, noisy_distances[idx], game_state)
                    positions.extend(possible_positions)
        return positions

    def get_possible_positions(self, agent_index, noisy_distance, game_state):
        """
        Returns possible positions of an opponent based on noisy distance.
        """
        my_pos = game_state.get_agent_position(agent_index)
        walls = game_state.get_walls()
        possible_positions = []

        for x in range(walls.width):
            for y in range(walls.height):
                if not walls[x][y]:
                    if self.distancer.get_distance(my_pos, (x, y)) == noisy_distance:
                        possible_positions.append((x, y))

        return possible_positions

    ########################
    # Escape Pathfinding   #
    ########################
    def find_safe_path(self, game_state):
        """
        Calculates a path to the safest location (e.g., home side or power-up).
        """
        my_pos = game_state.get_agent_position(self.index)
        home_positions = [
            pos for pos in self.get_home_positions(game_state)
            if self.distancer.get_distance(my_pos, pos) > 5
        ]
        return min(home_positions, key=lambda pos: self.distancer.get_distance(my_pos, pos), default=None)

    def get_home_positions(self, game_state):
        """
        Returns a list of positions on the agent's home side.
        """
        walls = game_state.get_walls()
        mid_x = walls.width // 2
        start_x = 0 if self.red else mid_x
        end_x = mid_x if self.red else walls.width
        home_positions = [
            (x, y) for x in range(start_x, end_x)
            for y in range(walls.height) if not walls[x][y]
        ]
        return home_positions

    ########################
    # Food Clustering      #
    ########################    
    def find_food_clusters(self, game_state):
        """
        Identifies high-density food clusters for offensive planning.
        """
        food_positions = self.get_food(game_state).as_list()
        clusters = []

        # Naive clustering: Group food by proximity
        for food in food_positions:
            for cluster in clusters:
                if any(self.distancer.get_distance(food, other) <= 2 for other in cluster):
                    cluster.append(food)
                    break
            else:
                clusters.append([food])

        # Sort clusters by size (descending)
        return sorted(clusters, key=len, reverse=True)

    ########################
    # Shared Data          #
    ########################
    def share_data(self, key, value):
        """
        Shares data between agents.
        """
        self.shared_data[key] = value

    def retrieve_data(self, key):
        """
        Retrieves shared data.
        """
        return self.shared_data.get(key, None)


import random
from queue import PriorityQueue
from itertools import count

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point, Stack


#################
# Team Creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
    """
    Creates a team of two agents for the capture-the-flag game.

    Args:
        first_index (int): Index of the first agent.
        second_index (int): Index of the second agent.
        is_red (bool): Indicates whether the team is red.
        first (str): Name of the first agent class.
        second (str): Name of the second agent class.

    Returns:
        list: A list containing two agent instances.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


#################
# Utility Methods #
#################

def find_nearest_open_space(game_state, target_position):
    """
    Finds the nearest open space to the target position.

    Args:
        game_state: The current game state.
        target_position (tuple): The target position (x, y).

    Returns:
        tuple: The position of the nearest open space.
    """
    stack = Stack()
    stack.push(target_position)
    visited = {target_position}

    while not stack.is_empty():
        current_position = stack.pop()
        if not is_surrounded_by_walls(game_state, current_position):
            return current_position

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_position = (current_position[0] + dx, current_position[1] + dy)
            if valid(game_state, new_position) and is_within_bounds_and_not_wall(game_state, new_position) and new_position not in visited:
                stack.push(new_position)
                visited.add(new_position)


def valid(game_state, position):
    """
    Checks if a position is valid.

    Args:
        game_state: The current game state.
        position (tuple): The position to check.

    Returns:
        bool: True if valid, False otherwise.
    """
    x, y = int(position[0]), int(position[1])
    return min(max(0, x), arena_width(game_state)) == x and min(max(0, y), arena_height(game_state)) == y


def is_surrounded_by_walls(game_state, position):
    """
    Checks if a position is surrounded by walls.

    Args:
        game_state: The current game state.
        position (tuple): The position to check.

    Returns:
        bool: True if surrounded by walls, False otherwise.
    """
    walls = game_state.get_walls()
    wall_count = sum(
        walls[position[0] + dx][position[1] + dy]
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
    )
    return wall_count > 1


def is_within_bounds_and_not_wall(game_state, position):
    """
    Checks if a position is within bounds and not a wall.

    Args:
        game_state: The current game state.
        position (tuple): The position to check.

    Returns:
        bool: True if valid, False otherwise.
    """
    walls = game_state.get_walls()
    x, y = int(position[0]), int(position[1])
    return not walls[x][y]


def arena_width(game_state):
    return game_state.get_walls().width

def arena_height(game_state):
    return game_state.get_walls().height
######################
# ReflexCaptureAgent #
######################

class ReflexCaptureAgent(CaptureAgent):
    """
    Base class for reflex agents that choose score-maximizing actions.
    """

    def __init__(self, index, time_for_computing=0.001):
        super().__init__(index, time_for_computing)
        self.start_position = None
        self.is_returning_home = False
        self.escape_mode = False
        self.escape_target = None

    def register_initial_state(self, game_state):
        """
        Registers the initial state of the agent.

        Args:
            game_state: The initial game state.
        """
        self.start_position = game_state.get_agent_position(self.index)
        super().register_initial_state(game_state)

    def get_next_game_state(self, game_state, action):
        """
        Simulates the next game state after an action.

        Args:
            game_state: The current game state.
            action (str): The action to take.

        Returns:
            GameState: The resulting game state.
        """
        next_state = game_state.generate_successor(self.index, action)
        if next_state.get_agent_position(self.index) != nearest_point(next_state.get_agent_position(self.index)):
            return next_state.generate_successor(self.index, action)
        return next_state

    def calculate_distances_with_positions(self, game_state, positions):
        return [(self.get_maze_distance(self.get_my_position(game_state), pos), pos) for pos in positions]
    

    def get_enemy_pacmen_positions(self, game_state):
        enemy_states = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        return [a.get_position() for a in enemy_states if a.is_pacman and a.get_position() is not None]
    

    def get_enemy_ghost_positions(self, game_state):
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]

        scared_ghosts = [a.get_position() for a in enemies if not a.is_pacman and a.scared_timer > 5 and a.get_position() is not None]
        ghosts = [a.get_position() for a in enemies if not a.is_pacman and a.scared_timer <= 5 and a.get_position() is not None]
        return ghosts, scared_ghosts
    
    def calculate_distances_to(self, game_state, positions):
        return [self.get_maze_distance(self.get_my_position(game_state), pos) for pos in positions]
    
    def get_my_position(self, game_state):
        return game_state.get_agent_state(self.index).get_position()

    def num_carrying(self, game_state):
        return game_state.get_agent_state(self.index).num_carrying

    
    def get_edge_home_cells(self, game_state, enemy_home=False):
        if enemy_home:
            w = arena_width(game_state) // 2 - (0 if self.red else 1)
        else:
            w = arena_width(game_state) // 2 - (1 if self.red else 0)
        home = []
        for h in range(1, arena_height(game_state) - 1):
            if is_within_bounds_and_not_wall(game_state, (w, h)):
                home.append((w, h))
        return home
    

    def is_pacman(self, game_state):
        state = game_state.get_agent_state(self.index)
        return state.is_pacman
    
    def is_leading(self, game_state):
        """
        Checks if the agent's team is leading.

        Args:
            game_state: The current game state.

        Returns:
            bool: True if leading, False otherwise.
        """
        return game_state.data.score if self.red else -game_state.data.score > 0

    def is_scared(self, game_state):
        """
        Checks if the agent is scared.

        Args:
            game_state: The current game state.

        Returns:
            bool: True if scared, False otherwise.
        """
        return game_state.get_agent_state(self.index).scared_timer > 0

    def get_enemy_positions(self, game_state):
        """
        Gets the positions of enemy agents.

        Args:
            game_state: The current game state.

        Returns:
            list: List of enemy positions.
        """
        return [
            state.get_position()
            for state in [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            if state.get_position() is not None
        ]

    def get_food_positions(self, game_state):
        """
        Retrieves the positions of food.

        Args:
            game_state: The current game state.

        Returns:
            list: List of food positions.
        """
        return self.get_food(game_state).as_list()

    
    def astar_pathfinding(self, game_state, target_position, excluded_positions=None):
        """
        Finds the optimal path to a target using A* search.

        Args:
            game_state: The current game state.
            target_position (tuple): The target position.
            excluded_positions (set): Optional positions to exclude.

        Returns:
            str: The first action in the optimal path.
        """
        priority_queue = PriorityQueue()
        visited = {self.start_position}
        if excluded_positions:
            visited.update(excluded_positions)

        counter = count()
        self._add_to_priority_queue(priority_queue, game_state, 0, None, None, target_position, counter)

        while not priority_queue.empty():
            previous, current_state, path_cost = self._extract_from_priority_queue(priority_queue)
            current_position = self.get_my_position(current_state)
            #print(previous)
            if current_position == target_position:
                #print(current_position, "\n", target_position)
                return self._extract_first_action(previous)

            for action in current_state.get_legal_actions(self.index):
                next_state = self.get_next_game_state(current_state, action)
                new_position = self.get_my_position(next_state)
                if new_position not in visited:
                    visited.add(new_position)
                    self._add_to_priority_queue(priority_queue, next_state, path_cost + 1, previous, action, target_position, counter)

        return "Stop"

    def _add_to_priority_queue(self, queue, game_state, cost, prev, action, target, counter):
        """
        Adds a state to the priority queue with its score.

        Args:
            queue (PriorityQueue): The priority queue.
            game_state: The game state.
            cost (int): The current path cost.
            prev (dict): The previous state.
            action (str): The action taken.
            target (tuple): The target position.
            counter (count): A counter for tie-breaking.
        """
        state_data = {
            "game_state": game_state,
            "cost": cost,
            "previous": prev,
            "action": action
        }
        #print(target)
        priority = self._compute_heuristic(self.get_my_position(game_state), target)
        queue.put((priority, next(counter), state_data))

    def _compute_heuristic(self, my_pos, target):
        #print(my_pos, target, distance_traveled)
        my_score = self.get_maze_distance(my_pos, target)
        return my_score
    
    def _extract_from_priority_queue(self, queue):
        """
        Extracts the top state from the priority queue.

        Args:
            queue (PriorityQueue): The priority queue.

        Returns:
            tuple: The previous state, current state, and cost.
        """
        _, _, state_data = queue.get()
        return state_data, state_data["game_state"], state_data["cost"]

    def _extract_first_action(self, state):
        """
        Extracts the first action from a chain of states.

        Args:
            state (dict): The current state data.

        Returns:
            str: The first action.
        """
        while state["previous"]["previous"]:
            state = state["previous"]
        return state["action"]


class OffensiveAgent(ReflexCaptureAgent):
    """
    Offensive agent focused on collecting food and avoiding threats.
    """
    """ def __init__(self):
        self.is_attacker = True """

    def choose_action(self, game_state):
        """
        Decides the best action for the offensive agent based on game state.

        Args:
            game_state: The current game state.

        Returns:
            str: The chosen action.
        """
        if self.should_panic(game_state):
            return "Stop"

        targets, excluded_positions = self.determine_targets(game_state)
        return self.find_best_action(game_state, targets, excluded_positions)

    def should_panic(self, game_state):
        """
        Determines if the agent should stop due to being in a bad state.

        Args:
            game_state: The current game state.

        Returns:
            bool: True if panic is necessary, False otherwise.
        """
        my_position = self.get_my_position(game_state)
        enemy_start = game_state.get_walls().width - 2 if self.red else 1
        return my_position[0] == enemy_start

    def determine_targets(self, game_state):
        """
        Identifies priority targets based on the game state.

        Args:
            game_state: The current game state.

        Returns:
            tuple: A list of target positions and excluded positions.
        """
        # Agent's current position
        my_position = self.get_my_position(game_state) 
        # Food positions coordinates closer to the agent
        food_positions = self.get_food_positions(game_state) 
        food_positions = sorted(food_positions, key=lambda pos: self.get_maze_distance(my_position, pos))
        # Number of food pieces carried
        carrying_food = self.num_carrying(game_state) 
        # Closest home edge positions to the agent without considering walls 
        home_positions = self.get_edge_home_cells(game_state) 
        home_positions = sorted(home_positions, key=lambda pos: self.get_maze_distance(my_position, pos))
        # Power capsule positions
        capsule_positions = self.get_capsules(game_state)
        # Visible ghost positions
        ghosts, _ = self.get_enemy_ghost_positions(game_state) 

        # Return home if carrying enough food or time is running out
        if self.should_return_home(game_state, carrying_food, len(food_positions)):
            return home_positions, []

        # Avoid deadlocks
        if self.escape_mode and my_position == self.escape_target:
            self.escape_mode = False

         # Prioritize capsules if they can be safely reached without interception
        if self.is_pacman(game_state) and ghosts:
            if capsule_positions:
                reachable_capsules = []
                for capsule in capsule_positions:
                    if self.is_path_safe(my_position, capsule, ghosts):
                        reachable_capsules.append(capsule)

                if reachable_capsules:
                    return sorted(reachable_capsules, key=lambda pos: self.get_maze_distance(my_position, pos)), []

        # Decide between returning home or collecting more food based on distance and risk
            min_home_distance = self.get_maze_distance(my_position, home_positions[0])
            if carrying_food > 0:
                min_ghost_distance = min(self.calculate_distances_to(game_state, ghosts))
                if min_ghost_distance < 3:  # Immediate danger threshold
                    self.is_returning_home = True
                    return home_positions, []

            # Collect food on the way home if it is along the path and time allows
            ghost_distance = min(self.get_maze_distance(ghost, home_positions[0]) for ghost in ghosts)
            food_on_path = self.food_along_safe_path(my_position, food_positions, home_positions[0], ghost_distance)
            if food_on_path:
                return food_on_path, []

            # If no food on path, prioritize home
            return home_positions, []

        # Default to food if no other conditions are met
        return food_positions, []
    

    def is_path_safe(self, start, target, ghosts):
        """
        Checks if the path to a target is safe (ghosts cannot intercept).

        Args:
            game_state: The current game state.
            start (tuple): Starting position (e.g., agent's position).
            target (tuple): Target position (e.g., capsule, food, home).
            ghosts (list): List of ghost positions.

        Returns:
            bool: True if the path is safe, False otherwise.
        """
        agent_distance = self.get_maze_distance(start, target)

        # Check if any ghost can reach the target faster
        ghost_distance = min(self.get_maze_distance(ghost, target) for ghost in ghosts)
        if ghost_distance <= agent_distance:
            return False
        return True
    
    def food_along_safe_path(self, my_position, food_positions, home_position, ghost_distance):
        """
        Identifies food positions that are along the path to home and safe to collect.

        Args:
            game_state: The current game state.
            food_positions (list): List of food positions closer to the agent.
            home_position (tuple): Position of the closest home cell.
            ghosts (list): List of ghost positions.

        Returns:
            list: A list of food positions that are safe to collect.
        """
        safe_food = []

        for food in food_positions:
            food_distance = self.get_maze_distance(my_position, food)
            food_home_distance = self.get_maze_distance(food, home_position)
            if food_distance + food_home_distance < ghost_distance:
                safe_food.append(food)
            
        return safe_food


    def should_return_home(self, game_state, carrying_food, food_left):
        """
        Checks if the agent should return home.

        Args:
            game_state: The current game state.
            carrying_food (int): Number of food pieces being carried.
            food_left (int): Number of food pieces remaining on the map.

        Returns:
            bool: True if returning home is a priority, False otherwise.
        """
        return food_left == 2 or (carrying_food and game_state.data.timeleft < 100)  # 100 means 25 moves for our attacker

    def find_best_action2(self, game_state, targets, excluded_positions):
        """
        Finds the best action to reach the given targets.

        Args:
            game_state: The current game state.
            targets (list): List of target positions.
            excluded_positions (list): List of positions to avoid.

        Returns:
            str: The chosen action.
        """
        ghosts, _ = self.get_enemy_ghost_positions(game_state)
        my_position = self.get_my_position(game_state)
        ghost_distances = self.calculate_distances_to(game_state, ghosts)

        for target in targets:
            open_space = find_nearest_open_space(game_state, target)
            ghost_distance = min(ghost_distances) if ghost_distances else float('inf')
            distance_to_target = self.get_maze_distance(my_position, open_space)

            # Safe path logic
            if distance_to_target * 2 + 1 < ghost_distance:
                return self.astar_pathfinding(game_state, target, excluded_positions)

        # Escape to safety if ghosts are close
        if ghosts and min(ghost_distances) <= 5:
            self.escape_mode = True
            self.escape_target = self.get_escape_position(game_state)
            return self.astar_pathfinding(game_state, self.escape_target, excluded_positions)

        # Default to the last target if no better options exist
        return self.astar_pathfinding(game_state, targets[-1], excluded_positions)
    
    def find_best_action(self, game_state, targets, excluded_positions):
        """
        Finds the best action to reach the given targets.

        Args:
            game_state: The current game state.
            targets (list): List of target positions.
            excluded_positions (list): List of positions to avoid.

        Returns:
            str: The chosen action.
        """

        # Default to the last target if no better options exist
        print("Attacker")
        return self.astar_pathfinding(game_state, targets[0], excluded_positions)

'''
class DefensiveAgent(ReflexCaptureAgent):
    """
    Defensive agent focused on protecting the home side and intercepting invaders.
    """
    """ def __init__(self):
        self.is_defender = True """

    def choose_action(self, game_state):
        """
        Decides the best action for the defensive agent based on game state.

        Args:
            game_state: The current game state.

        Returns:
            str: The chosen action.
        """
        targets, excluded_positions = self.determine_targets(game_state)
        if isinstance(targets, tuple): 
            print(excluded_positions)
            targets = targets[0]
            excluded_positions.append(targets[1])
        return self.find_best_action(game_state, targets, excluded_positions)

    def determine_targets(self, game_state):
        """
        Identifies priority targets for defense based on game state.

        Args:
            game_state: The current game state.

        Returns:
            tuple: A list of target positions and excluded positions.
        """
        my_position = self.get_my_position(game_state)
        home_positions = self.get_edge_home_cells(game_state) 
        excluded_positions = self.get_edge_home_cells(game_state, enemy_home=True)
        home_positions = sorted(home_positions, key=lambda pos: self.get_maze_distance(my_position, pos))
        if game_state.data.timeleft > 1060:
            return [home_positions[len(home_positions) // 2 - 1]], excluded_positions
        
        invader_positions = self.get_enemy_pacmen_positions(game_state)
        print(invader_positions)
        # Prioritize invaders
        if invader_positions:
            a = sorted(self.calculate_distances_with_positions(game_state, invader_positions))[:1], excluded_positions
            print(a)
            return a

        ghosts, scared_ghosts = self.get_enemy_ghost_positions(game_state)
        print("ghosts", ghosts)
        # Guard against enemy ghosts near home
        if ghosts or scared_ghosts:
            return sorted(self.calculate_distances_with_positions(game_state, ghosts + scared_ghosts))[:1], excluded_positions

        # Patrol home edges as a fallback
        return [random.choice(home_positions)], excluded_positions

    def find_best_action(self, game_state, targets, excluded_positions):
        """
        Finds the best action to reach the given targets.

        Args:
            game_state: The current game state.
            targets (list): List of target positions.
            excluded_positions (list): List of positions to avoid.

        Returns:
            str: The chosen action.
        """
        print("Defender:")
        print(targets[0])
        return self.astar_pathfinding(game_state, targets[0], excluded_positions)
'''


class DefensiveAgent(ReflexCaptureAgent):
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

        my_position = self.get_my_position(game_state)
        home_positions = self.get_edge_home_cells(game_state) 
        excluded_positions = self.get_edge_home_cells(game_state, enemy_home=True)
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
            if is_within_bounds_and_not_wall(game_state, curr_cell):
                return curr_cell

    def choose_best_action_for_target(self, game_state, targets, excluded_positions):
        return self.Astar(game_state, targets[0], excluded_positions)
    

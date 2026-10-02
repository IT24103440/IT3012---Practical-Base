import random
from collections import deque
import heapq
import math


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        return random.choice(self.actions_pool)


class SearchAgent:
    """Goal-based agent using BFS, DFS, UCS, or A*."""

    def __init__(self):
        self.plan = []

        # Practical 04:
        # Use A* as the active search algorithm
        self.active_algo = 'AStar'

    def get_neighbors(self, state, grid_size, walls):
        x, y = state
        width, height = grid_size

        # These directions MUST match execute_action()
        moves = [
            ((x, y + 1), 'Up'),
            ((x, y - 1), 'Down'),
            ((x - 1, y), 'Left'),
            ((x + 1, y), 'Right')
        ]

        neighbors = []

        for new_state, action in moves:
            nx, ny = new_state

            # Check grid boundaries
            if 0 <= nx < width and 0 <= ny < height:

                # Check walls
                if new_state not in walls:
                    neighbors.append((new_state, action))

        return neighbors

    # =========================================================
    # HEURISTIC FUNCTIONS
    # =========================================================

    def manhattan_distance(self, pos, goal):
        """
        Calculate Manhattan distance between two positions.

        Formula:
        h(n) = |x1 - x2| + |y1 - y2|
        """

        x1, y1 = pos
        x2, y2 = goal

        return abs(x1 - x2) + abs(y1 - y2)

    def euclidean_distance(self, pos, goal):
        """
        Calculate Euclidean distance between two positions.

        Formula:
        h(n) = sqrt((x1 - x2)^2 + (y1 - y2)^2)
        """

        x1, y1 = pos
        x2, y2 = goal

        return math.sqrt(
            (x1 - x2) ** 2 +
            (y1 - y2) ** 2
        )

    # =========================================================
    # BFS
    # =========================================================

    def bfs_search(self, start, goal, grid_size, walls):

        frontier = deque()
        frontier.append((start, []))

        reached = {start}

        while frontier:

            state, path = frontier.popleft()

            # Goal reached
            if state == goal:
                return path

            for next_state, action in self.get_neighbors(
                state,
                grid_size,
                walls
            ):

                if next_state not in reached:

                    reached.add(next_state)

                    frontier.append(
                        (
                            next_state,
                            path + [action]
                        )
                    )

        # No path found
        return []

    # =========================================================
    # DFS
    # =========================================================

    def dfs_search(self, start, goal, grid_size, walls):

        frontier = []
        frontier.append((start, []))

        reached = {start}

        while frontier:

            state, path = frontier.pop()

            # Goal reached
            if state == goal:
                return path

            for next_state, action in self.get_neighbors(
                state,
                grid_size,
                walls
            ):

                if next_state not in reached:

                    reached.add(next_state)

                    frontier.append(
                        (
                            next_state,
                            path + [action]
                        )
                    )

        # No path found
        return []

    # =========================================================
    # UCS
    # =========================================================

    def ucs_search(self, start, goal, grid_size, walls):

        frontier = []

        # (cost, state, path)
        heapq.heappush(
            frontier,
            (0, start, [])
        )

        reached = {
            start: 0
        }

        while frontier:

            cost, state, path = heapq.heappop(frontier)

            # Goal reached
            if state == goal:
                return path

            for next_state, action in self.get_neighbors(
                state,
                grid_size,
                walls
            ):

                # Every movement has cost 1
                new_cost = cost + 1

                if (
                    next_state not in reached
                    or new_cost < reached[next_state]
                ):

                    reached[next_state] = new_cost

                    heapq.heappush(
                        frontier,
                        (
                            new_cost,
                            next_state,
                            path + [action]
                        )
                    )

        # No path found
        return []

    # =========================================================
    # A* SEARCH
    # =========================================================

    def astar_search(
        self,
        start_pos,
        goal_pos,
        walls,
        grid_size,
        heuristic_type='manhattan'
    ):
        """
        A* Search Algorithm.

        f(n) = g(n) + h(n)

        g(n) = path cost from start to current node
        h(n) = estimated cost from current node to goal
        f(n) = estimated total cost
        """

        # Priority queue
        frontier = []

        # Set of already explored states
        reached_states = set()

        # -----------------------------------------------------
        # Calculate heuristic for the starting position
        # -----------------------------------------------------

        if heuristic_type == 'euclidean':

            h_cost = self.euclidean_distance(
                start_pos,
                goal_pos
            )

        else:

            h_cost = self.manhattan_distance(
                start_pos,
                goal_pos
            )

        # Starting node:
        # (f_cost, g_cost, current_pos, path_taken)
        heapq.heappush(
            frontier,
            (
                h_cost,
                0,
                start_pos,
                []
            )
        )

        # -----------------------------------------------------
        # A* main loop
        # -----------------------------------------------------

        while frontier:

            (
                f_cost,
                g_cost,
                current_pos,
                path_taken
            ) = heapq.heappop(frontier)

            # -------------------------------------------------
            # Goal test
            # -------------------------------------------------

            if current_pos == goal_pos:
                return path_taken

            # Skip if already explored
            if current_pos in reached_states:
                continue

            # Mark current state as reached
            reached_states.add(current_pos)

            # -------------------------------------------------
            # Expand neighboring nodes
            # -------------------------------------------------

            for next_pos, action in self.get_neighbors(
                current_pos,
                grid_size,
                walls
            ):

                if next_pos not in reached_states:

                    # Every movement costs 1
                    new_g_cost = g_cost + 1

                    # -----------------------------------------
                    # Calculate h(n)
                    # -----------------------------------------

                    if heuristic_type == 'euclidean':

                        new_h_cost = self.euclidean_distance(
                            next_pos,
                            goal_pos
                        )

                    else:

                        new_h_cost = self.manhattan_distance(
                            next_pos,
                            goal_pos
                        )

                    # -----------------------------------------
                    # Calculate f(n)
                    # f(n) = g(n) + h(n)
                    # -----------------------------------------

                    new_f_cost = (
                        new_g_cost +
                        new_h_cost
                    )

                    # Add neighbor to priority queue
                    heapq.heappush(
                        frontier,
                        (
                            new_f_cost,
                            new_g_cost,
                            next_pos,
                            path_taken + [action]
                        )
                    )

        # No path found
        return []

    # =========================================================
    # SENSE AND ACT
    # =========================================================

    def sense_and_act(self, percept: dict) -> str:

        # Create a new plan when the current plan is empty
        if not self.plan:

            start = tuple(percept['agent_pos'])

            foods = percept['all_food']

            # No food remaining
            if not foods:
                return None

            grid_size = percept['grid_size']

            walls = set(percept['walls'])

            # -------------------------------------------------
            # Find the closest food using Manhattan distance
            # -------------------------------------------------

            goal = min(
                foods,
                key=lambda food:
                self.manhattan_distance(
                    start,
                    food
                )
            )

            # -------------------------------------------------
            # Select search algorithm
            # -------------------------------------------------

            if self.active_algo == 'BFS':

                self.plan = self.bfs_search(
                    start,
                    goal,
                    grid_size,
                    walls
                )

            elif self.active_algo == 'DFS':

                self.plan = self.dfs_search(
                    start,
                    goal,
                    grid_size,
                    walls
                )

            elif self.active_algo == 'UCS':

                self.plan = self.ucs_search(
                    start,
                    goal,
                    grid_size,
                    walls
                )

            elif self.active_algo == 'AStar':

                self.plan = self.astar_search(
                    start,
                    goal,
                    walls,
                    grid_size,
                    heuristic_type='manhattan'
                )

            else:

                print(
                    "Invalid search algorithm:",
                    self.active_algo
                )

                return None

        # -----------------------------------------------------
        # Execute the next action from the plan
        # -----------------------------------------------------

        if self.plan:
            return self.plan.pop(0)

        # No path available
        return None


# =============================================================
# PRACTICAL 04 - HEURISTIC TESTING CHECKPOINT
# =============================================================

if __name__ == "__main__":

    agent = SearchAgent()

    start_position = (0, 0)
    goal_position = (3, 4)

    manhattan_result = agent.manhattan_distance(
        start_position,
        goal_position
    )

    euclidean_result = agent.euclidean_distance(
        start_position,
        goal_position
    )

    print(
        "Manhattan Distance:",
        manhattan_result
    )

    print(
        "Euclidean Distance:",
        euclidean_result
    )
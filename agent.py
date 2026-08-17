# agent.py

import random
from collections import deque
import heapq


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        pos = percept['agent_pos']
        return random.choice(self.actions_pool)


class SearchAgent:
    """Goal-based agent using BFS, DFS, or UCS."""

    def __init__(self):
        self.plan = []
        self.active_algo = 'BFS'

    def get_neighbors(self, state, grid_size, walls):
        x, y = state
        width, height = grid_size

        moves = [
            ((x, y - 1), 'Up'),
            ((x, y + 1), 'Down'),
            ((x - 1, y), 'Left'),
            ((x + 1, y), 'Right')
        ]

        neighbors = []

        for new_state, action in moves:
            nx, ny = new_state

            if 0 <= nx < width and 0 <= ny < height:
                if new_state not in walls:
                    neighbors.append((new_state, action))

        return neighbors

    # -------------------------
    # BFS
    # -------------------------
    def bfs_search(self, start, goal, grid_size, walls):

        frontier = deque()
        frontier.append((start, []))

        reached = {start}

        while frontier:
            state, path = frontier.popleft()

            if state == goal:
                return path

            for next_state, action in self.get_neighbors(
                state, grid_size, walls
            ):
                if next_state not in reached:

                    reached.add(next_state)

                    frontier.append(
                        (next_state, path + [action])
                    )

        return []

    # -------------------------
    # DFS
    # -------------------------
    def dfs_search(self, start, goal, grid_size, walls):

        frontier = []
        frontier.append((start, []))

        reached = {start}

        while frontier:
            state, path = frontier.pop()

            if state == goal:
                return path

            for next_state, action in self.get_neighbors(
                state, grid_size, walls
            ):
                if next_state not in reached:

                    reached.add(next_state)

                    frontier.append(
                        (next_state, path + [action])
                    )

        return []

    # -------------------------
    # UCS
    # -------------------------
    def ucs_search(self, start, goal, grid_size, walls):

        frontier = []

        # cost, state, path
        heapq.heappush(
            frontier,
            (0, start, [])
        )

        reached = {start: 0}

        while frontier:

            cost, state, path = heapq.heappop(frontier)

            if state == goal:
                return path

            for next_state, action in self.get_neighbors(
                state, grid_size, walls
            ):

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

        return []

    # -------------------------
    # Sense and Act
    # -------------------------
    def sense_and_act(self, percept: dict) -> str:

        # Create a new plan if current plan is empty
        if not self.plan:

            start = tuple(percept['agent_pos'])

            foods = percept['all_food']

            # No food remaining
            if not foods:
                return None

            # Find closest food using Manhattan distance
            goal = min(
                foods,
                key=lambda food:
                abs(food[0] - start[0]) +
                abs(food[1] - start[1])
            )

            grid_size = percept['grid_size']
            walls = set(percept['walls'])

            # Select search algorithm
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

        # Execute the next action
        if self.plan:
            return self.plan.pop(0)

        return None

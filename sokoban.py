"""
Sokoban search and planning

Premise: The same Sokoban problem/environment can be solved using different search algorithms:
    Breadth-First Search, Uniform-Cost Search, and A* Search
"""

from collections import deque
from dataclasses import astuple, dataclass

import heapq
import textwrap
import time

BOARD = """
########
###   ##
#.@$  ##
### $.##
#.##$ ##
# # . ##
#$ *$$.#
#   .  #
########
"""

MOVEMENTS = [
        ("Up", (-1, 0)),
        ("Down", (1, 0)),
        ("Left", (0, -1)),
        ("Right", (0, 1)),
]


@dataclass(frozen=True)
class SokobanState:
    player: tuple
    crates: frozenset

@dataclass
class SearchNode:
    state: SokobanState
    parent: object
    action: str
    path_cost: int
    depth: int

@dataclass
class SearchResult:
    algorithm: str
    solution: bool
    actions: list
    states: list
    path_cost: int
    generated_nodes: int
    expanded_nodes: int
    max_frontier: int
    runtime: float

@dataclass(frozen=True)
class StateSymbols:
    wall: str = "#"
    goals: str = "."
    crate: str = "$"
    player: str = "@"
    crate_on_goal: str = "*"
    player_on_goal: str = "+"



class Sokoban:
    def __init__(self, board_str, symbols=None):
        self.symbols = StateSymbols() if symbols is None else symbols
        wall, goal, crate, player_symbol, crate_on_goal, player_on_goal = astuple(self.symbols)

        self.board_lines = self._clean_board(board_str)
        self.rows = len(self.board_lines)
        self.cols = max(len(line) for line in self.board_lines)
        self.walls = set()
        self.goals = set()

        crates = set()
        player = None

        for row, line in enumerate(self.board_lines):
            padded_line = line.ljust(self.cols)
            for col, symbol in enumerate(padded_line):
                position = (row, col)
                if symbol == wall:
                    self.walls.add(position)
                elif symbol == goal:
                    self.goals.add(position)
                elif symbol == crate:
                    crates.add(position)
                elif symbol == player_symbol:
                    player = position
                elif symbol == crate_on_goal:
                    crates.add(position)
                    self.goals.add(position)
                elif symbol == player_on_goal:
                    player = position
                    self.goals.add(position)

        if player is None:
            raise ValueError("The board must contain one player marked with '" + player_symbol + "'.")
        elif not crates:
            raise ValueError("The board must contain at least one crate marked with '" + crate + "'.")
        elif len(crates) != len(self.goals):
            raise ValueError("The board should contain the same number of crates and goals.")

        self.initial_state = SokobanState(player, frozenset(crates))

    def _clean_board(self, board_text):
        lines = textwrap.dedent(board_text).strip("\n").splitlines()
        if not lines:
            raise ValueError("The board cannot be empty.")
        return [line.rstrip("\n") for line in lines]

    def actions(self, state):
        legal_actions = []

        for direction_name, (row_change, col_change) in MOVEMENTS:
            next_position = (
                state.player[0] + row_change,
                state.player[1] + col_change,
            )

            if next_position in state.crates:
                crate_next_position = (
                    next_position[0] + row_change,
                    next_position[1] + col_change,
                )
                if self._is_free(crate_next_position, state.crates):
                    legal_actions.append("Push " + direction_name)
            elif self._is_free(next_position, state.crates):
                legal_actions.append("Move " + direction_name)

        return legal_actions

    def result(self, state, action):
        action_type, direction_name = action.split()
        row_change, col_change = self._direction_change(direction_name)
        next_player = (
            state.player[0] + row_change,
            state.player[1] + col_change,
        )

        if action_type == "Move":
            return SokobanState(next_player, state.crates)

        if action_type == "Push":
            next_crate = (
                next_player[0] + row_change,
                next_player[1] + col_change,
            )
            new_crates = set(state.crates)
            new_crates.remove(next_player)
            new_crates.add(next_crate)
            return SokobanState(next_player, frozenset(new_crates))

        raise ValueError("Unknown action: " + action)

    def is_goal(self, state):
        return state.crates == self.goals

    def step_cost(self, state, action, next_state):
        if action.startswith("Push"):
            return 2
        return 1

    def _is_free(self, position, crates):
        return self._in_bounds(position) and position not in self.walls and position not in crates

    def _in_bounds(self, position):
        row, col = position
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _direction_change(self, direction_name):
        for name, (row_change, col_change) in MOVEMENTS:
            if name == direction_name:
                return row_change, col_change
        raise ValueError("Unknown direction: " + direction_name)


""" search algorithm implementations for sokoban game space """
class Search:
    def __init__(self, on_expand=None):
        self.on_expand = on_expand

    def bfs(self, sokoban: Sokoban):
        start_time = time.perf_counter()
        start_node = SearchNode(sokoban.initial_state, None, None, 0, 0)

        frontier = deque([start_node])
        reached = {sokoban.initial_state}
        
        max_frontier, expanded, generated = 1, 0, 1

        while frontier:
            node = frontier.popleft()
            if sokoban.is_goal(node.state):
                return self._build_result("BFS", node, start_time, generated, expanded, max_frontier)

            expanded += 1
            for act in sokoban.actions(node.state):
                next_state = sokoban.result(node.state, act)
                if next_state not in reached:
                    reached.add(next_state)
                    next_cost = node.path_cost + sokoban.step_cost(node.state, act, next_state)
                    child = SearchNode(next_state, node, act, next_cost, node.depth + 1)

                    frontier.append(child)
                
                    generated += 1

            max_frontier = max(max_frontier, len(frontier))
            if self.on_expand:
                self.on_expand(node, len(frontier), generated, expanded, max_frontier)

        return self._build_result("BFS", None, start_time, generated, expanded, max_frontier)


    def ufs(self, sokoban: Sokoban):
        start_time = time.perf_counter()
        start_node = SearchNode(sokoban.initial_state, None, None, 0, 0)
        best_cost = {sokoban.initial_state: 0}

        frontier = [(0, 0, start_node)]

        max_frontier, expanded, generated, counter = 1, 0, 1, 1

        while frontier:
            current_cost, unused_order, node = heapq.heappop(frontier)
            if current_cost != best_cost[node.state]:
                continue
            elif sokoban.is_goal(node.state):
                return self._build_result("UCS", node, start_time, generated, expanded, max_frontier)

            expanded += 1
            for act in sokoban.actions(node.state):
            
                new_state = sokoban.result(node.state, act)
                new_cost = node.path_cost + sokoban.step_cost(node.state, act, new_state)

                if (new_state not in best_cost) or new_cost < best_cost[new_state]:
                    best_cost[new_state] = new_cost

                    child = SearchNode(new_state, node, act, new_cost, node.depth + 1)
                    heapq.heappush(frontier, (new_cost, counter, child))
                    
                    generated += 1
                    counter += 1

            max_frontier = max(max_frontier, len(frontier))
            if self.on_expand:
                self.on_expand(node, len(frontier), generated, expanded, max_frontier)

        return self._build_result("UCS", None, start_time, generated, expanded, max_frontier)
                 
                                               
    def a_star(self, sokoban: Sokoban):
        # Sokoban heuristic function; 
        def heuristic_function(state, sokoban: Sokoban):
            total = 0
            for crate in state.crates:
                closest_goal = min(
                    abs(crate[0] - goal[0]) + abs(crate[1] - goal[1])
                    for goal in sokoban.goals
                )
                total += closest_goal * 2
            return total

        start_time = time.perf_counter()
        start_node = SearchNode(sokoban.initial_state, None, None, 0, 0)
        start_priority = heuristic_function(sokoban.initial_state, sokoban)

        frontier = [(start_priority, 0, start_node)]
        best_cost = {sokoban.initial_state: 0}

        max_frontier, expanded, generated, counter = 1, 0, 1, 1


        while frontier:
            _, _, node = heapq.heappop(frontier)
            if node.path_cost != best_cost[node.state]:
                continue
            elif sokoban.is_goal(node.state):
                return self._build_result("A*", node, start_time, generated, expanded, max_frontier)

            expanded += 1
            for act in sokoban.actions(node.state):
                new_state = sokoban.result(node.state, act)
                new_cost = node.path_cost + sokoban.step_cost(node.state, act, new_state)

                if (new_state not in best_cost) or new_cost < best_cost[new_state]:
                    best_cost[new_state] = new_cost
                    child = SearchNode(new_state, node, act, new_cost, node.depth + 1)
                    
                    priority = new_cost + heuristic_function(new_state, sokoban)
                    heapq.heappush(frontier, (priority, counter, child))

                    counter += 1
                    generated += 1

            max_frontier = max(max_frontier, len(frontier))
            if self.on_expand:
                self.on_expand(node, len(frontier), generated, expanded, max_frontier)

        return self._build_result("A*", None, start_time, generated, expanded, max_frontier)

    def _build_result(self, name, goal_node, start_time, generated, expanded, max_frontier):
        def _get_solution(goal_node):
            actions, states = [], []
            
            node = goal_node
            while node is not None:
                states.append(node.state)
                if node.action is not None:
                    actions.append(node.action)
                node = node.parent
            actions.reverse()
            states.reverse()
            return actions, states

        solution: bool = False if goal_node is None else True
        actions, states = ([], []) if goal_node is None else _get_solution(goal_node)
        runtime = time.perf_counter() - start_time

        if solution:
            return SearchResult(
                name,
                solution,
                actions,
                states,
                goal_node.path_cost,
                generated,
                expanded,
                max_frontier,
                runtime,
            )

        return SearchResult(
            name,
            solution,
            actions,
            states,
            0,
            generated,
            expanded,
            max_frontier,
            runtime,
        )


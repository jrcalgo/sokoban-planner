from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

try:
    import pygame
except ModuleNotFoundError:
    pygame = None

from sokoban import BOARD, Search, SearchResult, Sokoban, SokobanState


WINDOW_SIZE = (1180, 760)
FPS = 60
STEPS_PER_SECOND = 3
MAX_RECORDED_STEPS = 50000
MAX_GRAPH_POINTS = 2000
MAX_SELECTED_METRICS = 8
PANEL_WIDTH = 340
PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"

BACKGROUND = (31, 34, 42)
PANEL = (43, 47, 58)
TEXT = (238, 238, 238)
MUTED = (172, 178, 190)
ACCENT = (95, 175, 255)
DISABLED = (86, 91, 103)
FLOOR = (222, 214, 173)
WALL = (118, 118, 118)
WALL_DARK = (70, 70, 70)
GOAL = (213, 150, 134)
CRATE = (245, 151, 36)
CRATE_ON_GOAL = (94, 188, 112)
PLAYER = (54, 129, 226)
PLAYER_FACE = (255, 232, 98)
TRAIL = (118, 181, 255)
GRAPH_COLORS = [
    (95, 175, 255),
    (128, 220, 150),
    (255, 198, 92),
    (255, 146, 126),
    (185, 148, 255),
    (88, 211, 199),
    (245, 151, 36),
    (220, 220, 120),
]


ALGORITHMS = [
    ("Breadth-First Search", "bfs"),
    ("Uniform-Cost Search", "ufs"),
    ("A* Search", "a_star"),
]


class SearchStopped(Exception):
    pass


@dataclass(frozen=True)
class StepFrame:
    state: SokobanState
    action: str
    path_cost: int
    depth: int
    frontier_size: int = 0
    generated_nodes: int = 0
    expanded_nodes: int = 0
    max_frontier: int = 0


@dataclass(frozen=True)
class AnalyticsSample:
    expansion_index: int
    elapsed: float
    generated_nodes: int
    expanded_nodes: int
    frontier_size: int
    max_frontier: int
    depth: int
    path_cost: int
    crates_on_goals: int
    move_actions: int
    push_actions: int


@dataclass(frozen=True)
class TrailSegment:
    start: tuple
    end: tuple
    kind: str
    direction: str
    arrow: str
    current: bool = False


@dataclass(frozen=True)
class MetricDefinition:
    key: str
    label: str
    getter: object
    formatter: object


def format_int(value):
    return f"{int(round(value)):,}"


def format_float(value):
    return f"{value:.2f}"


def format_seconds(value):
    return f"{value:.2f}s"


METRICS = [
    MetricDefinition("elapsed", "Runtime", lambda sample: sample.elapsed, format_seconds),
    MetricDefinition("expanded_nodes", "Expanded Nodes", lambda sample: sample.expanded_nodes, format_int),
    MetricDefinition("generated_nodes", "Generated Nodes", lambda sample: sample.generated_nodes, format_int),
    MetricDefinition("frontier_size", "Frontier Size", lambda sample: sample.frontier_size, format_int),
    MetricDefinition("max_frontier", "Maximum Frontier", lambda sample: sample.max_frontier, format_int),
    MetricDefinition("depth", "Node Depth", lambda sample: sample.depth, format_int),
    MetricDefinition("path_cost", "Path Cost", lambda sample: sample.path_cost, format_int),
    MetricDefinition("crates_on_goals", "Crates on Goals", lambda sample: sample.crates_on_goals, format_int),
    MetricDefinition("move_actions", "Move Actions", lambda sample: sample.move_actions, format_int),
    MetricDefinition("push_actions", "Push Actions", lambda sample: sample.push_actions, format_int),
    MetricDefinition(
        "generated_ratio",
        "Generated / Expanded",
        lambda sample: sample.generated_nodes / max(1, sample.expanded_nodes),
        format_float,
    ),
    MetricDefinition("frontier_growth", "Frontier Growth", lambda sample: sample.frontier_size - 1, format_int),
]
METRIC_BY_KEY = {metric.key: metric for metric in METRICS}
DEFAULT_ANALYTICS_METRICS = [
    "expanded_nodes",
    "generated_nodes",
    "frontier_size",
    "max_frontier",
    "depth",
    "path_cost",
    "crates_on_goals",
    "elapsed",
]

DIRECTION_INFO = {
    "Up": ((-1, 0), "↑"),
    "Down": ((1, 0), "↓"),
    "Left": ((0, -1), "←"),
    "Right": ((0, 1), "→"),
}


def parse_action_direction(action):
    if not action or action == "Start":
        return "Start", "", (0, 0), ""
    parts = action.split()
    if len(parts) != 2:
        return action, "", (0, 0), ""
    kind, direction = parts
    delta, arrow = DIRECTION_INFO.get(direction, ((0, 0), ""))
    return kind, direction, delta, arrow


def format_action_label(action):
    kind, direction, unused_delta, arrow = parse_action_direction(action)
    if kind == "Start":
        return "Start", "", ""
    return kind, arrow, direction


class AlgorithmRun:
    def __init__(self, game, label, method_name):
        self.game = game
        self.label = label
        self.method_name = method_name
        self.lock = threading.Lock()
        self.cancel_event = threading.Event()
        self.thread = None
        self.search_steps = deque(maxlen=MAX_RECORDED_STEPS)
        self.dropped_search_steps = 0
        self.solution_steps = []
        self.result = None
        self.error = None
        self.status = "ready"
        self.start_time = None
        self.end_time = None
        self.generated_nodes = 1
        self.expanded_nodes = 0
        self.max_frontier = 1
        self.frontier_size = 1
        self.analytics_samples = []
        self.analytics_stride = 1
        self.analytics_seen = 0
        self.move_actions = 0
        self.push_actions = 0

    def start(self):
        if self.thread is not None:
            return
        self.status = "searching"
        self.start_time = time.perf_counter()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def cancel(self):
        self.cancel_event.set()

    def _run(self):
        try:
            search = Search(on_expand=self._record_expansion)
            result = getattr(search, self.method_name)(self.game)
            with self.lock:
                self.result = result
                self.solution_steps = self._build_solution_steps(result)
                self.status = "solved" if result.solution else "no solution"
                self.end_time = time.perf_counter()
            self._write_completed_log(result)
        except SearchStopped:
            with self.lock:
                self.status = "cancelled"
                self.end_time = time.perf_counter()
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
                self.status = "error"
                self.end_time = time.perf_counter()

    def _write_completed_log(self, result):
        try:
            log_data = result_to_log(
                self.game,
                self.label,
                self.method_name,
                result,
                board_hash(self.game),
            )
            write_cli_log(unique_log_path(self.method_name), log_data)
        except OSError:
            pass

    def _record_expansion(self, node, frontier_size, generated, expanded, max_frontier):
        if self.cancel_event.is_set():
            raise SearchStopped()

        if node.action and node.action.startswith("Move"):
            self.move_actions += 1
        elif node.action and node.action.startswith("Push"):
            self.push_actions += 1

        frame = StepFrame(
            state=node.state,
            action=node.action or "Start",
            path_cost=node.path_cost,
            depth=node.depth,
            frontier_size=frontier_size,
            generated_nodes=generated,
            expanded_nodes=expanded,
            max_frontier=max_frontier,
        )
        with self.lock:
            if len(self.search_steps) == self.search_steps.maxlen:
                self.dropped_search_steps += 1
            self.search_steps.append(frame)
            self.frontier_size = frontier_size
            self.generated_nodes = generated
            self.expanded_nodes = expanded
            self.max_frontier = max_frontier
            self._record_analytics(node, frontier_size, generated, expanded, max_frontier)

    def _record_analytics(self, node, frontier_size, generated, expanded, max_frontier):
        self.analytics_seen += 1
        if self.analytics_seen % self.analytics_stride != 0:
            return

        elapsed = 0 if self.start_time is None else time.perf_counter() - self.start_time
        sample = AnalyticsSample(
            expansion_index=expanded,
            elapsed=elapsed,
            generated_nodes=generated,
            expanded_nodes=expanded,
            frontier_size=frontier_size,
            max_frontier=max_frontier,
            depth=node.depth,
            path_cost=node.path_cost,
            crates_on_goals=len(node.state.crates & self.game.goals),
            move_actions=self.move_actions,
            push_actions=self.push_actions,
        )
        self.analytics_samples.append(sample)
        if len(self.analytics_samples) > MAX_GRAPH_POINTS:
            self.analytics_stride *= 2
            self.analytics_samples = self.analytics_samples[::2]

    def _build_solution_steps(self, result: SearchResult):
        if not result.solution:
            return []

        steps = [
            StepFrame(
                state=result.states[0],
                action="Start",
                path_cost=0,
                depth=0,
                generated_nodes=result.generated_nodes,
                expanded_nodes=result.expanded_nodes,
                max_frontier=result.max_frontier,
            )
        ]
        path_cost = 0
        for index, action in enumerate(result.actions, start=1):
            previous_state = result.states[index - 1]
            state = result.states[index]
            path_cost += self.game.step_cost(previous_state, action, state)
            steps.append(
                StepFrame(
                    state=state,
                    action=action,
                    path_cost=path_cost,
                    depth=index,
                    generated_nodes=result.generated_nodes,
                    expanded_nodes=result.expanded_nodes,
                    max_frontier=result.max_frontier,
                )
            )
        return steps

    def search_length(self):
        with self.lock:
            return len(self.search_steps)

    def solution_length(self):
        with self.lock:
            return len(self.solution_steps)

    def search_step_at(self, index):
        with self.lock:
            if not self.search_steps:
                return None
            index = max(0, min(index, len(self.search_steps) - 1))
            return list(self.search_steps)[index]

    def solution_step_at(self, index):
        with self.lock:
            if not self.solution_steps:
                return None
            index = max(0, min(index, len(self.solution_steps) - 1))
            return self.solution_steps[index]

    def snapshot(self):
        with self.lock:
            result = self.result
            elapsed_end = self.end_time or time.perf_counter()
            elapsed = 0 if self.start_time is None else elapsed_end - self.start_time
            return {
                "status": self.status,
                "elapsed": elapsed,
                "error": self.error,
                "generated_nodes": self.generated_nodes,
                "expanded_nodes": self.expanded_nodes,
                "max_frontier": self.max_frontier,
                "frontier_size": self.frontier_size,
                "dropped_search_steps": self.dropped_search_steps,
                "recorded_search_steps": len(self.search_steps),
                "solution_length": len(self.solution_steps),
                "solution_found": bool(result and result.solution),
                "path_cost": 0 if result is None else result.path_cost,
                "actions": [] if result is None else list(result.actions),
            }

    def snapshot_analytics(self):
        with self.lock:
            result = self.result
            elapsed_end = self.end_time or time.perf_counter()
            elapsed = 0 if self.start_time is None else elapsed_end - self.start_time
            return {
                "label": self.label,
                "method_name": self.method_name,
                "status": self.status,
                "elapsed": elapsed,
                "samples": list(self.analytics_samples),
                "stride": self.analytics_stride,
                "sample_count": len(self.analytics_samples),
                "observed_expansions": self.analytics_seen,
                "solution_found": bool(result and result.solution),
                "solution_length": 0 if result is None else len(result.actions),
                "path_cost": 0 if result is None else result.path_cost,
                "generated_nodes": self.generated_nodes,
                "expanded_nodes": self.expanded_nodes,
                "max_frontier": self.max_frontier,
            }


class Stepper:
    def __init__(self):
        self.index = 0
        self.playing = False
        self.timer = 0.0

    def back(self):
        self.index = max(0, self.index - 1)
        self.playing = False

    def forward(self, length):
        if length:
            self.index = min(length - 1, self.index + 1)
        self.playing = False

    def play(self):
        self.playing = True

    def pause(self):
        self.playing = False

    def restart(self):
        self.index = 0
        self.timer = 0.0
        self.playing = False

    def clamp(self, length):
        if length <= 0:
            self.index = 0
        else:
            self.index = max(0, min(self.index, length - 1))

    def update(self, dt, length, live=False):
        self.clamp(length)
        if not self.playing or length <= 0:
            return

        self.timer += dt
        interval = 1.0 / STEPS_PER_SECOND
        while self.timer >= interval:
            self.timer -= interval
            if self.index + 1 < length:
                self.index += 1
            elif live:
                break
            else:
                self.playing = False
                break


class Button:
    def __init__(self, rect, label, callback, enabled=lambda: True, selected=lambda: False):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.callback = callback
        self.enabled = enabled
        self.selected = selected

    def handle_event(self, event):
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return False
        if self.enabled() and self.rect.collidepoint(event.pos):
            self.callback()
            return True
        return False

    def draw(self, surface, font):
        enabled = self.enabled()
        selected = self.selected()
        mouse_over = self.rect.collidepoint(pygame.mouse.get_pos())
        color = ACCENT if enabled and mouse_over else (64, 70, 84)
        if selected:
            color = (75, 100, 132)
        if not enabled:
            color = DISABLED
        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        border = ACCENT if selected else (18, 20, 26)
        pygame.draw.rect(surface, border, self.rect, 2, border_radius=8)
        label = font.render(self.label, True, TEXT if enabled else MUTED)
        surface.blit(label, label.get_rect(center=self.rect.center))


def draw_text(surface, font, text, pos, color=TEXT):
    rendered = font.render(str(text), True, color)
    surface.blit(rendered, pos)
    return rendered.get_height()


def draw_wrapped(surface, font, text, rect, color=TEXT, line_gap=4):
    words = str(text).split()
    line = ""
    y = rect.top
    for word in words:
        candidate = word if not line else line + " " + word
        if font.size(candidate)[0] <= rect.width:
            line = candidate
            continue
        if line:
            surface.blit(font.render(line, True, color), (rect.left, y))
            y += font.get_height() + line_gap
        line = word
    if line and y < rect.bottom:
        surface.blit(font.render(line, True, color), (rect.left, y))
        y += font.get_height() + line_gap
    return y


def metric_values(samples, metric):
    return [float(metric.getter(sample)) for sample in samples]


def metric_stats(values):
    if not values:
        return {"min": 0, "max": 0, "average": 0, "latest": 0, "count": 0}
    return {
        "min": min(values),
        "max": max(values),
        "average": sum(values) / len(values),
        "latest": values[-1],
        "count": len(values),
    }


def draw_line_graph(surface, rect, values, color):
    rect = pygame.Rect(rect)
    pygame.draw.rect(surface, (25, 28, 36), rect, border_radius=6)
    pygame.draw.rect(surface, (64, 70, 84), rect, 1, border_radius=6)

    if not values:
        return

    graph_rect = rect.inflate(-16, -14)
    min_value = min(values)
    max_value = max(values)
    value_range = max(max_value - min_value, 1.0)

    points = []
    count = len(values)
    for index, value in enumerate(values):
        x_ratio = 0 if count == 1 else index / (count - 1)
        y_ratio = (value - min_value) / value_range
        x = graph_rect.left + x_ratio * graph_rect.width
        y = graph_rect.bottom - y_ratio * graph_rect.height
        points.append((int(x), int(y)))

    if len(points) == 1:
        pygame.draw.circle(surface, color, points[0], 3)
    else:
        pygame.draw.lines(surface, color, False, points, 2)


def draw_metric_card(surface, rect, title, values, formatter, color, app):
    rect = pygame.Rect(rect)
    pygame.draw.rect(surface, PANEL, rect, border_radius=10)
    pygame.draw.rect(surface, (18, 20, 26), rect, 1, border_radius=10)

    x = rect.left + 12
    y = rect.top + 10
    title_width = rect.width - 24
    clipped_title = title
    while app.small_font.size(clipped_title)[0] > title_width and len(clipped_title) > 4:
        clipped_title = clipped_title[:-4] + "..."
    surface.blit(app.small_font.render(clipped_title, True, TEXT), (x, y))
    y += app.small_font.get_height() + 6

    graph_rect = pygame.Rect(x, y, rect.width - 24, max(42, rect.height - 92))
    draw_line_graph(surface, graph_rect, values, color)
    y = graph_rect.bottom + 8

    stats = metric_stats(values)
    if not values:
        draw_text(surface, app.small_font, "No samples yet", (x, y), MUTED)
        return

    latest = formatter(stats["latest"])
    minimum = formatter(stats["min"])
    maximum = formatter(stats["max"])
    average = formatter(stats["average"])
    draw_text(surface, app.small_font, f"Latest: {latest}", (x, y), TEXT)
    draw_text(surface, app.small_font, f"Min {minimum}  Max {maximum}", (x, y + 20), MUTED)
    draw_text(surface, app.small_font, f"Avg {average}  N {stats['count']:,}", (x, y + 40), MUTED)


def tile_center(origin_x, origin_y, tile, position):
    row, col = position
    return (origin_x + col * tile + tile // 2, origin_y + row * tile + tile // 2)


def draw_arrow_segment(surface, start, end, color, width, arrow_size):
    if start == end:
        return

    pygame.draw.line(surface, color, start, end, width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    tip = end
    base = (end[0] - ux * arrow_size, end[1] - uy * arrow_size)
    left = (base[0] + px * arrow_size * 0.55, base[1] + py * arrow_size * 0.55)
    right = (base[0] - px * arrow_size * 0.55, base[1] - py * arrow_size * 0.55)
    pygame.draw.polygon(surface, color, [tip, left, right])


def draw_trail_segments(surface, trail, origin_x, origin_y, tile):
    for segment in trail or ():
        if not isinstance(segment, TrailSegment):
            continue

        start = tile_center(origin_x, origin_y, tile, segment.start)
        end = tile_center(origin_x, origin_y, tile, segment.end)
        if segment.kind == "Push":
            color = (246, 170, 74) if segment.current else (176, 112, 49)
            width = max(3, tile // 8) if segment.current else max(2, tile // 10)
        else:
            color = (117, 192, 255) if segment.current else (83, 133, 188)
            width = max(2, tile // 11) if segment.current else max(1, tile // 14)

        draw_arrow_segment(surface, start, end, color, width, max(6, tile // 5 if segment.current else tile // 6))


def action_color(kind, active=False):
    if kind == "Push":
        return (246, 170, 74) if active else (176, 112, 49)
    return (117, 192, 255) if active else (83, 133, 188)


def draw_action_badge(surface, rect, kind, font, active=False):
    rect = pygame.Rect(rect)
    color = action_color(kind, active)
    pygame.draw.rect(surface, color, rect, border_radius=6)
    pygame.draw.rect(surface, (18, 20, 26), rect, 1, border_radius=6)
    label = font.render(kind, True, (18, 20, 26))
    surface.blit(label, label.get_rect(center=rect.center))


def draw_solution_legend(surface, rect, app):
    rect = pygame.Rect(rect)
    y = rect.top + 2
    draw_action_badge(surface, pygame.Rect(rect.left, y, 54, 22), "Move", app.small_font)
    draw_text(surface, app.small_font, "player path", (rect.left + 62, y + 2), MUTED)
    draw_action_badge(surface, pygame.Rect(rect.left + 160, y, 54, 22), "Push", app.small_font)
    draw_text(surface, app.small_font, "crate push", (rect.left + 222, y + 2), MUTED)


def draw_board(surface, game, state, area, trail=()):
    area = pygame.Rect(area)
    tile = max(8, min(area.width // game.cols, area.height // game.rows))
    width = tile * game.cols
    height = tile * game.rows
    origin_x = area.left + (area.width - width) // 2
    origin_y = area.top + (area.height - height) // 2

    pygame.draw.rect(surface, (22, 24, 30), (origin_x - 8, origin_y - 8, width + 16, height + 16), border_radius=12)

    for row in range(game.rows):
        for col in range(game.cols):
            pos = (row, col)
            rect = pygame.Rect(origin_x + col * tile, origin_y + row * tile, tile, tile)
            if pos in game.walls:
                pygame.draw.rect(surface, WALL, rect)
                pygame.draw.line(surface, WALL_DARK, rect.topleft, rect.bottomright, 1)
                pygame.draw.line(surface, WALL_DARK, rect.topright, rect.bottomleft, 1)
            else:
                pygame.draw.rect(surface, FLOOR, rect)
                if pos in game.goals:
                    pygame.draw.circle(surface, GOAL, rect.center, max(4, tile // 5))

    draw_trail_segments(surface, trail, origin_x, origin_y, tile)

    for crate in state.crates:
        row, col = crate
        rect = pygame.Rect(origin_x + col * tile, origin_y + row * tile, tile, tile).inflate(-tile // 5, -tile // 5)
        color = CRATE_ON_GOAL if crate in game.goals else CRATE
        pygame.draw.rect(surface, color, rect, border_radius=max(2, tile // 12))
        pygame.draw.rect(surface, (25, 18, 10), rect, max(1, tile // 18), border_radius=max(2, tile // 12))
        pygame.draw.line(surface, (25, 18, 10), rect.topleft, rect.bottomright, max(1, tile // 24))
        pygame.draw.line(surface, (25, 18, 10), rect.topright, rect.bottomleft, max(1, tile // 24))

    row, col = state.player
    center = (origin_x + col * tile + tile // 2, origin_y + row * tile + tile // 2)
    pygame.draw.circle(surface, PLAYER, center, max(6, tile // 3))
    pygame.draw.circle(surface, PLAYER_FACE, (center[0], center[1] - tile // 8), max(4, tile // 5))
    pygame.draw.circle(surface, (15, 15, 15), (center[0] - tile // 13, center[1] - tile // 7), max(1, tile // 35))
    pygame.draw.circle(surface, (15, 15, 15), (center[0] + tile // 13, center[1] - tile // 7), max(1, tile // 35))


def make_solution_trail(solution_steps, index):
    if not solution_steps or index <= 0:
        return ()
    segments = []
    last_index = min(index, len(solution_steps) - 1)
    for step_index in range(1, last_index + 1):
        action = solution_steps[step_index].action
        kind, direction, unused_delta, arrow = parse_action_direction(action)
        segments.append(
            TrailSegment(
                start=solution_steps[step_index - 1].state.player,
                end=solution_steps[step_index].state.player,
                kind=kind,
                direction=direction,
                arrow=arrow,
                current=step_index == last_index,
            )
        )
    return tuple(segments)


class AnalyticsPanel:
    def __init__(self, app, selected_metrics=None):
        self.app = app
        if selected_metrics is None:
            self.selected_metrics = list(DEFAULT_ANALYTICS_METRICS)
        else:
            self.selected_metrics = list(selected_metrics)
        self.buttons = []
        self.hint = ""

    def toggle_metric(self, key):
        if key in self.selected_metrics:
            self.selected_metrics.remove(key)
            self.hint = ""
        elif len(self.selected_metrics) < MAX_SELECTED_METRICS:
            self.selected_metrics.append(key)
            self.hint = ""
        else:
            self.hint = f"Up to {MAX_SELECTED_METRICS} metrics can be selected."

    def handle_event(self, event):
        for button in self.buttons:
            if button.handle_event(event):
                return True
        return False

    def draw(self, surface, rect, analytics, title_prefix=""):
        rect = pygame.Rect(rect)
        self.buttons = []
        pygame.draw.rect(surface, BACKGROUND, rect)

        menu_rect = pygame.Rect(rect.left, rect.top, 230, rect.height)
        graph_rect = pygame.Rect(menu_rect.right + 18, rect.top, rect.width - menu_rect.width - 18, rect.height)

        pygame.draw.rect(surface, PANEL, menu_rect, border_radius=12)
        y = menu_rect.top + 16
        draw_text(surface, self.app.header_font, "Metrics", (menu_rect.left + 14, y))
        y += 42

        for metric in METRICS:
            selected = lambda key=metric.key: key in self.selected_metrics
            button = Button(
                pygame.Rect(menu_rect.left + 14, y, menu_rect.width - 28, 34),
                metric.label,
                lambda key=metric.key: self.toggle_metric(key),
                selected=selected,
            )
            self.buttons.append(button)
            button.draw(surface, self.app.small_font)
            y += 40

        if self.hint:
            draw_wrapped(
                surface,
                self.app.small_font,
                self.hint,
                pygame.Rect(menu_rect.left + 14, menu_rect.bottom - 70, menu_rect.width - 28, 60),
                (255, 198, 92),
            )

        samples = analytics["samples"]
        if not samples:
            draw_wrapped(
                surface,
                self.app.header_font,
                "Waiting for search data...",
                pygame.Rect(graph_rect.left + 20, graph_rect.top + 20, graph_rect.width - 40, 120),
                MUTED,
            )
            return

        selected = [METRIC_BY_KEY[key] for key in self.selected_metrics if key in METRIC_BY_KEY][:MAX_SELECTED_METRICS]
        if not selected:
            draw_wrapped(
                surface,
                self.app.header_font,
                "Select metrics from the left menu.",
                pygame.Rect(graph_rect.left + 20, graph_rect.top + 20, graph_rect.width - 40, 120),
                MUTED,
            )
            return

        gap = 12
        cols = 2
        rows = 4
        card_width = (graph_rect.width - gap) // cols
        card_height = (graph_rect.height - gap * (rows - 1)) // rows

        for index, metric in enumerate(selected):
            col = index % cols
            row = index // cols
            card = pygame.Rect(
                graph_rect.left + col * (card_width + gap),
                graph_rect.top + row * (card_height + gap),
                card_width,
                card_height,
            )
            title = title_prefix + metric.label
            values = metric_values(samples, metric)
            draw_metric_card(surface, card, title, values, metric.formatter, GRAPH_COLORS[index % len(GRAPH_COLORS)], self.app)


class MenuScreen:
    def __init__(self, app):
        self.app = app
        self.buttons = []

    def open_run(self, label, method_name):
        run = self.app.runs.get(method_name)
        if run is None or run.snapshot()["status"] in {"cancelled", "error"}:
            run = AlgorithmRun(self.app.game, label, method_name)
            self.app.runs[method_name] = run
        self.app.screen = RunScreen(self.app, run)

    def open_all_analytics(self):
        self.app.screen = AllAnalyticsScreen(self.app)

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.app.running = False
        for button in self.buttons:
            if button.handle_event(event):
                return

    def update(self, dt):
        pass

    def draw(self, surface):
        surface.fill(BACKGROUND)
        title = self.app.title_font.render("Sokoban Search Visualizer", True, TEXT)
        surface.blit(title, (40, 32))
        draw_text(surface, self.app.small_font, "Choose an algorithm. Searches keep running until they solve, fail, or you go back.", (42, 86), MUTED)

        draw_board(surface, self.app.game, self.app.game.initial_state, pygame.Rect(40, 130, 560, 500))

        self.buttons = []
        y = 150
        for label, method_name in ALGORITHMS:
            rect = pygame.Rect(660, y, 360, 54)
            self.buttons.append(Button(rect, label, lambda l=label, m=method_name: self.open_run(l, m)))
            y += 82

            run = self.app.runs.get(method_name)
            status = "Not run yet"
            if run is not None:
                status = format_run_status(run.snapshot())
            draw_wrapped(surface, self.app.small_font, status, pygame.Rect(670, y - 22, 420, 42), MUTED)

        self.buttons.append(Button(pygame.Rect(660, y + 16, 250, 48), "All Analytics", self.open_all_analytics))
        self.buttons.append(Button(pygame.Rect(660, y + 76, 200, 48), "Quit", lambda: setattr(self.app, "running", False)))
        for button in self.buttons:
            button.draw(surface, self.app.button_font)

        draw_text(surface, self.app.small_font, "Visual search history is a rolling buffer; old expansion frames are discarded to save memory.", (42, 684), MUTED)


class AllAnalyticsScreen:
    def __init__(self, app):
        self.app = app
        self.buttons = []
        self.expanded = set()
        self.selected_metrics = []
        self.hint = ""

    def back_to_menu(self):
        self.app.screen = MenuScreen(self.app)

    def available_runs(self):
        runs = []
        for label, method_name in ALGORITHMS:
            run = self.app.runs.get(method_name)
            if run is None:
                continue
            analytics = run.snapshot_analytics()
            if analytics["samples"]:
                runs.append((label, method_name, run, analytics))
        return runs

    def toggle_algorithm(self, method_name):
        if method_name in self.expanded:
            self.expanded.remove(method_name)
        else:
            self.expanded.add(method_name)

    def toggle_metric(self, method_name, metric_key):
        pair = (method_name, metric_key)
        if pair in self.selected_metrics:
            self.selected_metrics.remove(pair)
            self.hint = ""
        elif len(self.selected_metrics) < MAX_SELECTED_METRICS:
            self.selected_metrics.append(pair)
            self.hint = ""
        else:
            self.hint = f"Up to {MAX_SELECTED_METRICS} metrics can be selected."

    def prune_selected(self):
        available_methods = {method_name for _, method_name, _, _ in self.available_runs()}
        self.selected_metrics = [pair for pair in self.selected_metrics if pair[0] in available_methods]

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.app.running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.back_to_menu()

        for button in self.buttons:
            if button.handle_event(event):
                return

    def update(self, dt):
        pass

    def draw(self, surface):
        surface.fill(BACKGROUND)
        self.prune_selected()
        runs = self.available_runs()
        if runs and not self.expanded:
            self.expanded.add(runs[0][1])

        title = self.app.title_font.render("All Analytics", True, TEXT)
        surface.blit(title, (28, 24))
        draw_text(surface, self.app.small_font, "Compare collected metrics across algorithms that have been run.", (30, 78), MUTED)

        self.buttons = [
            Button(pygame.Rect(WINDOW_SIZE[0] - 178, 28, 150, 42), "Back to Menu", self.back_to_menu)
        ]
        self.buttons[0].draw(surface, self.app.button_font)

        menu_rect = pygame.Rect(28, 110, 300, WINDOW_SIZE[1] - 140)
        graph_rect = pygame.Rect(menu_rect.right + 18, 110, WINDOW_SIZE[0] - menu_rect.right - 46, WINDOW_SIZE[1] - 140)
        pygame.draw.rect(surface, PANEL, menu_rect, border_radius=12)

        if not runs:
            draw_wrapped(
                surface,
                self.app.header_font,
                "Run an algorithm first to collect analytics.",
                pygame.Rect(graph_rect.left + 20, graph_rect.top + 20, graph_rect.width - 40, 120),
                MUTED,
            )
            return

        y = menu_rect.top + 16
        draw_text(surface, self.app.header_font, "Algorithms", (menu_rect.left + 14, y))
        y += 42

        for label, method_name, run, analytics in runs:
            if y > menu_rect.bottom - 42:
                draw_text(surface, self.app.small_font, "Collapse menus to see more.", (menu_rect.left + 14, menu_rect.bottom - 28), (255, 198, 92))
                break

            button = Button(
                pygame.Rect(menu_rect.left + 14, y, menu_rect.width - 28, 34),
                ("v " if method_name in self.expanded else "> ") + label,
                lambda method=method_name: self.toggle_algorithm(method),
                selected=lambda method=method_name: method in self.expanded,
            )
            self.buttons.append(button)
            button.draw(surface, self.app.small_font)
            y += 40

            if method_name not in self.expanded:
                continue

            for metric in METRICS:
                if y > menu_rect.bottom - 42:
                    draw_text(surface, self.app.small_font, "Collapse menus to see more.", (menu_rect.left + 14, menu_rect.bottom - 28), (255, 198, 92))
                    break
                pair = (method_name, metric.key)
                metric_button = Button(
                    pygame.Rect(menu_rect.left + 34, y, menu_rect.width - 48, 30),
                    metric.label,
                    lambda method=method_name, key=metric.key: self.toggle_metric(method, key),
                    selected=lambda selected_pair=pair: selected_pair in self.selected_metrics,
                )
                self.buttons.append(metric_button)
                metric_button.draw(surface, self.app.small_font)
                y += 34
            y += 6

        if self.hint:
            draw_wrapped(
                surface,
                self.app.small_font,
                self.hint,
                pygame.Rect(menu_rect.left + 14, menu_rect.bottom - 72, menu_rect.width - 28, 60),
                (255, 198, 92),
            )

        analytics_by_method = {method_name: analytics for _, method_name, _, analytics in runs}
        label_by_method = {method_name: label for label, method_name, _, _ in runs}
        selected = self.selected_metrics[:MAX_SELECTED_METRICS]
        if not selected:
            draw_wrapped(
                surface,
                self.app.header_font,
                "Select metrics from an algorithm dropdown.",
                pygame.Rect(graph_rect.left + 20, graph_rect.top + 20, graph_rect.width - 40, 120),
                MUTED,
            )
            return

        gap = 12
        cols = 2
        rows = 4
        card_width = (graph_rect.width - gap) // cols
        card_height = (graph_rect.height - gap * (rows - 1)) // rows

        for index, (method_name, metric_key) in enumerate(selected):
            metric = METRIC_BY_KEY.get(metric_key)
            analytics = analytics_by_method.get(method_name)
            if metric is None or analytics is None:
                continue

            col = index % cols
            row = index // cols
            card = pygame.Rect(
                graph_rect.left + col * (card_width + gap),
                graph_rect.top + row * (card_height + gap),
                card_width,
                card_height,
            )
            title = f"{label_by_method[method_name]}: {metric.label}"
            values = metric_values(analytics["samples"], metric)
            draw_metric_card(surface, card, title, values, metric.formatter, GRAPH_COLORS[index % len(GRAPH_COLORS)], self.app)


class RunScreen:
    def __init__(self, app, run):
        self.app = app
        self.run = run
        self.tab = "search"
        self.steppers = {"search": Stepper(), "solution": Stepper()}
        self.analytics_panel = AnalyticsPanel(app)
        self.buttons = []
        self.run.start()

    def back_to_menu(self):
        if self.run.snapshot()["status"] == "searching":
            self.run.cancel()
        self.app.screen = MenuScreen(self.app)

    def switch_tab(self, tab):
        stepper = self.steppers.get(self.tab)
        if stepper:
            stepper.pause()
        self.tab = tab

    def active_length(self):
        if self.tab == "search":
            return self.run.search_length()
        if self.tab == "solution":
            return self.run.solution_length()
        return 0

    def active_stepper(self):
        return self.steppers.get(self.tab, self.steppers["search"])

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.app.running = False
        elif event.type == pygame.KEYDOWN:
            self.handle_key(event.key)

        if self.tab == "analytics" and self.analytics_panel.handle_event(event):
            return

        for button in self.buttons:
            if button.handle_event(event):
                return

    def handle_key(self, key):
        if key == pygame.K_ESCAPE:
            self.back_to_menu()
            return
        if key == pygame.K_TAB:
            order = ["search", "solution", "analytics"]
            self.switch_tab(order[(order.index(self.tab) + 1) % len(order)])
            return
        if self.tab == "analytics":
            return

        stepper = self.active_stepper()
        length = self.active_length()
        if key == pygame.K_LEFT:
            stepper.back()
        elif key == pygame.K_RIGHT:
            stepper.forward(length)
        elif key == pygame.K_SPACE:
            if stepper.playing:
                stepper.pause()
            else:
                stepper.play()
        elif key == pygame.K_r:
            stepper.restart()

    def update(self, dt):
        for tab, stepper in self.steppers.items():
            length = self.run.search_length() if tab == "search" else self.run.solution_length()
            live = tab == "search" and self.run.snapshot()["status"] == "searching"
            stepper.update(dt, length, live=live)

    def current_frame(self):
        stepper = self.active_stepper()
        if self.tab == "search":
            return self.run.search_step_at(stepper.index)
        return self.run.solution_step_at(stepper.index)

    def draw(self, surface):
        surface.fill(BACKGROUND)
        snapshot = self.run.snapshot()
        self.draw_top_bar(surface)

        if self.tab == "analytics":
            self.analytics_panel.draw(
                surface,
                pygame.Rect(28, 106, WINDOW_SIZE[0] - 56, WINDOW_SIZE[1] - 136),
                self.run.snapshot_analytics(),
            )
            return

        frame = self.current_frame()
        if frame is None:
            frame = StepFrame(self.app.game.initial_state, "Start", 0, 0)

        trail = ()
        if self.tab == "solution" and self.run.solution_length():
            steps = [self.run.solution_step_at(i) for i in range(self.run.solution_length())]
            trail = make_solution_trail(steps, self.active_stepper().index)

        board_area = pygame.Rect(28, 106, WINDOW_SIZE[0] - PANEL_WIDTH - 60, 552)
        draw_board(surface, self.app.game, frame.state, board_area, trail)

        panel_rect = pygame.Rect(WINDOW_SIZE[0] - PANEL_WIDTH - 20, 106, PANEL_WIDTH, 552)
        self.draw_stats(surface, panel_rect, snapshot, frame)
        self.draw_controls(surface, snapshot)

    def draw_top_bar(self, surface):
        title = self.app.title_font.render(self.run.label, True, TEXT)
        surface.blit(title, (28, 24))

        self.buttons = []
        self.buttons.append(Button(pygame.Rect(438, 28, 120, 42), "Search", lambda: self.switch_tab("search"), lambda: self.tab != "search"))
        self.buttons.append(Button(pygame.Rect(568, 28, 130, 42), "Solution", lambda: self.switch_tab("solution"), lambda: self.tab != "solution"))
        self.buttons.append(Button(pygame.Rect(708, 28, 130, 42), "Analytics", lambda: self.switch_tab("analytics"), lambda: self.tab != "analytics"))
        self.buttons.append(Button(pygame.Rect(WINDOW_SIZE[0] - 178, 28, 150, 42), "Back to Menu", self.back_to_menu))
        for button in self.buttons:
            button.draw(surface, self.app.button_font)

        if self.tab == "search":
            pygame.draw.rect(surface, ACCENT, pygame.Rect(438, 72, 120, 4))
        elif self.tab == "solution":
            pygame.draw.rect(surface, ACCENT, pygame.Rect(568, 72, 130, 4))
        else:
            pygame.draw.rect(surface, ACCENT, pygame.Rect(708, 72, 130, 4))

    def draw_stats(self, surface, rect, snapshot, frame):
        pygame.draw.rect(surface, PANEL, rect, border_radius=12)
        x = rect.left + 18
        y = rect.top + 18

        y += draw_text(surface, self.app.header_font, "Run Stats", (x, y)) + 12
        status = snapshot["status"].replace("_", " ").title()
        y += draw_text(surface, self.app.small_font, f"Status: {status}", (x, y), status_color(snapshot["status"])) + 6
        y += draw_text(surface, self.app.small_font, f"Runtime: {snapshot['elapsed']:.2f}s", (x, y), MUTED) + 6
        y += draw_text(surface, self.app.small_font, f"Expanded: {snapshot['expanded_nodes']:,}", (x, y), MUTED) + 6
        y += draw_text(surface, self.app.small_font, f"Generated: {snapshot['generated_nodes']:,}", (x, y), MUTED) + 6
        y += draw_text(surface, self.app.small_font, f"Max frontier: {snapshot['max_frontier']:,}", (x, y), MUTED) + 6
        y += draw_text(surface, self.app.small_font, f"Path cost: {snapshot['path_cost']}", (x, y), MUTED) + 6

        if snapshot["error"]:
            y += 10
            y = draw_wrapped(surface, self.app.small_font, "Error: " + snapshot["error"], pygame.Rect(x, y, rect.width - 36, 100), (255, 146, 126))

        y += 18
        y += draw_text(surface, self.app.header_font, "Current Frame", (x, y)) + 12
        if self.tab == "search":
            length = self.run.search_length()
            dropped = snapshot["dropped_search_steps"]
            global_index = dropped + self.steppers["search"].index + 1 if length else 0
            y += draw_text(surface, self.app.small_font, f"Expansion: {global_index:,}", (x, y), MUTED) + 6
            y += draw_text(surface, self.app.small_font, f"Recorded: {length:,}", (x, y), MUTED) + 6
            y += draw_text(surface, self.app.small_font, f"Discarded: {dropped:,}", (x, y), MUTED) + 6
            y += draw_text(surface, self.app.small_font, f"Frontier now: {frame.frontier_size:,}", (x, y), MUTED) + 6
        else:
            length = self.run.solution_length()
            step = self.steppers["solution"].index + 1 if length else 0
            y += draw_text(surface, self.app.small_font, f"Step: {step} of {length}", (x, y), MUTED) + 6
            crates_on_goals = len(frame.state.crates & self.app.game.goals)
            y += draw_text(surface, self.app.small_font, f"Crates on goals: {crates_on_goals}/{len(self.app.game.goals)}", (x, y), MUTED) + 6

        y += draw_text(surface, self.app.small_font, f"Depth: {frame.depth}", (x, y), MUTED) + 6
        y += draw_text(surface, self.app.small_font, f"Cost so far: {frame.path_cost}", (x, y), MUTED) + 6
        y += draw_text(surface, self.app.small_font, f"Last action: {frame.action}", (x, y), MUTED) + 14

        if self.tab == "solution":
            draw_solution_legend(surface, pygame.Rect(x, y, rect.width - 36, 28), self.app)
            y += 38
            self.draw_action_list(surface, pygame.Rect(x, y, rect.width - 36, rect.bottom - y - 16), snapshot)
        elif snapshot["dropped_search_steps"]:
            draw_wrapped(
                surface,
                self.app.small_font,
                "Older search frames were discarded from the visual history. The solver is still searching with its own internal frontier.",
                pygame.Rect(x, y, rect.width - 36, rect.bottom - y - 16),
                MUTED,
            )

    def draw_action_list(self, surface, rect, snapshot):
        actions = snapshot["actions"]
        if not actions:
            message = "Waiting for a solution to replay." if snapshot["status"] == "searching" else "No solution to replay."
            draw_wrapped(surface, self.app.small_font, message, rect, MUTED)
            return

        current = self.steppers["solution"].index - 1
        focus = max(0, current)
        y = rect.top
        row_height = 30
        max_rows = max(1, rect.height // row_height)
        first = max(0, min(focus - max_rows // 2, len(actions) - max_rows))
        for index in range(first, min(len(actions), first + max_rows)):
            action = actions[index]
            kind, arrow, direction = format_action_label(action)
            active = index == current
            row_rect = pygame.Rect(rect.left, y, rect.width, row_height - 3)
            if active:
                pygame.draw.rect(surface, (52, 68, 88), row_rect, border_radius=7)
                pygame.draw.rect(surface, ACCENT, pygame.Rect(row_rect.left, row_rect.top, 4, row_rect.height), border_radius=3)

            color = TEXT if active else MUTED
            arrow_label = self.app.header_font.render(arrow, True, action_color(kind, active))
            surface.blit(arrow_label, (rect.left + 12, y + 2))
            draw_action_badge(surface, pygame.Rect(rect.left + 42, y + 4, 54, 20), kind, self.app.small_font, active)
            action_text = f"{index + 1}. {direction}"
            surface.blit(self.app.small_font.render(action_text, True, color), (rect.left + 108, y + 6))
            y += row_height

    def draw_controls(self, surface, snapshot):
        y = WINDOW_SIZE[1] - 76
        length = self.active_length()
        stepper = self.active_stepper()
        solution_ready = self.tab == "search" or self.run.solution_length() > 0

        controls = [
            ("Restart", stepper.restart, lambda: length > 0 and solution_ready),
            ("< Back", stepper.back, lambda: length > 0 and stepper.index > 0 and solution_ready),
            ("Play", stepper.play, lambda: length > 0 and not stepper.playing and solution_ready),
            ("Pause", stepper.pause, lambda: stepper.playing),
            ("Forward >", lambda: stepper.forward(length), lambda: length > 0 and stepper.index + 1 < length and solution_ready),
        ]

        x = 28
        for label, callback, enabled in controls:
            button = Button(pygame.Rect(x, y, 126, 48), label, callback, enabled)
            self.buttons.append(button)
            button.draw(surface, self.app.button_font)
            x += 138


def format_board(game, state):
    rows = []
    for row in range(game.rows):
        symbols = []
        for col in range(game.cols):
            position = (row, col)
            if position in game.walls:
                symbol = "#"
            elif position == state.player and position in game.goals:
                symbol = "+"
            elif position == state.player:
                symbol = "@"
            elif position in state.crates and position in game.goals:
                symbol = "*"
            elif position in state.crates:
                symbol = "$"
            elif position in game.goals:
                symbol = "."
            else:
                symbol = " "
            symbols.append(symbol)
        rows.append("".join(symbols).rstrip())
    return "\n".join(rows)


def normalized_board_text(game):
    return "\n".join(line.rstrip() for line in game.board_lines)


def board_hash(game):
    return hashlib.sha256(normalized_board_text(game).encode("utf-8")).hexdigest()


def cli_log_path(method_name):
    names = {"bfs": "BFS.json", "ufs": "UCS.json", "a_star": "A_star.json"}
    return LOG_DIR / names[method_name]


def unique_log_path(method_name):
    path = cli_log_path(method_name)
    if not path.exists():
        return path
    index = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def write_cli_log(path, data):
    LOG_DIR.mkdir(exist_ok=True)
    with path.open("w", encoding="utf-8") as log_file:
        json.dump(data, log_file, indent=2)
        log_file.write("\n")


def result_to_log(game, label, method_name, result, current_board_hash):
    return {
        "algorithm_name": result.algorithm,
        "display_name": label,
        "method_name": method_name,
        "board_hash": current_board_hash,
        "board": normalized_board_text(game),
        "solution_found": result.solution,
        "solution_length": len(result.actions),
        "path_cost": result.path_cost,
        "generated_nodes": result.generated_nodes,
        "expanded_nodes": result.expanded_nodes,
        "max_frontier": result.max_frontier,
        "runtime": result.runtime,
        "actions": list(result.actions),
    }


def run_algorithm_for_cli(game, label, method_name):
    search = Search()
    result = getattr(search, method_name)(game)
    log_data = result_to_log(game, label, method_name, result, board_hash(game))
    path = unique_log_path(method_name)
    write_cli_log(path, log_data)

    print()
    print("=" * 64)
    print(label)
    print("=" * 64)
    print("Solution found:", "Yes" if log_data["solution_found"] else "No")
    print("Solution length:", log_data["solution_length"])
    print("Path cost:", log_data["path_cost"])
    print("Generated nodes:", f"{log_data['generated_nodes']:,}")
    print("Expanded nodes:", f"{log_data['expanded_nodes']:,}")
    print("Maximum frontier:", f"{log_data['max_frontier']:,}")
    print("Runtime: {:.6f} seconds".format(log_data["runtime"]))
    print("Log:", "wrote new log", f"({path})")
    print()
    return log_data


def print_cli_menu():
    print("Sokoban Search CLI")
    print("1. Breadth-First Search")
    print("2. Uniform-Cost Search")
    print("3. A* Search")
    print("4. Run all algorithms")
    print("5. Show initial board")
    print("0. Exit")


def run_cli():
    game = Sokoban(BOARD)
    print("Initial Board")
    print(format_board(game, game.initial_state))
    print()

    choices = {
        "1": [ALGORITHMS[0]],
        "2": [ALGORITHMS[1]],
        "3": [ALGORITHMS[2]],
        "4": ALGORITHMS,
    }

    while True:
        print_cli_menu()
        choice = input("Choose an option: ").strip()
        if choice == "0":
            print("Goodbye.")
            return
        if choice == "5":
            print()
            print("Initial Board")
            print(format_board(game, game.initial_state))
            print()
            continue
        if choice not in choices:
            print("Invalid choice. Please choose 0, 1, 2, 3, 4, or 5.")
            print()
            continue

        for label, method_name in choices[choice]:
            run_algorithm_for_cli(game, label, method_name)


class App:
    def __init__(self):
        if pygame is None:
            raise RuntimeError("pygame is required for GUI mode. Use --cli or install pygame.")
        pygame.init()
        pygame.display.set_caption("Sokoban Search Visualizer")
        self.surface = pygame.display.set_mode(WINDOW_SIZE)
        self.clock = pygame.time.Clock()
        self.running = True
        self.game = Sokoban(BOARD)
        self.runs = {}
        self.title_font = pygame.font.SysFont("arial", 34, bold=True)
        self.header_font = pygame.font.SysFont("arial", 23, bold=True)
        self.button_font = pygame.font.SysFont("arial", 18, bold=True)
        self.small_font = pygame.font.SysFont("arial", 17)
        self.screen = MenuScreen(self)

    def run(self):
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                self.screen.handle_event(event)
            self.screen.update(dt)
            self.screen.draw(self.surface)
            pygame.display.flip()

        for run in self.runs.values():
            run.cancel()
        pygame.quit()


def status_color(status):
    if status == "solved":
        return (128, 220, 150)
    if status in {"error", "cancelled"}:
        return (255, 146, 126)
    if status == "searching":
        return ACCENT
    return MUTED


def format_run_status(snapshot):
    status = snapshot["status"]
    if status == "solved":
        return "Solved: {} actions, cost {}, {:,} expanded, {:.1f}s".format(
            max(0, snapshot["solution_length"] - 1),
            snapshot["path_cost"],
            snapshot["expanded_nodes"],
            snapshot["elapsed"],
        )
    if status == "searching":
        return "Searching: {:,} expanded, {:,} frontier, {:.1f}s".format(
            snapshot["expanded_nodes"], snapshot["frontier_size"], snapshot["elapsed"]
        )
    if status == "no solution":
        return "No solution: {:,} expanded, {:.1f}s".format(snapshot["expanded_nodes"], snapshot["elapsed"])
    if status == "cancelled":
        return "Cancelled after {:,} expansions.".format(snapshot["expanded_nodes"])
    if status == "error":
        return "Error: " + (snapshot["error"] or "unknown error")
    return "Ready"


def main():
    if "--cli" in sys.argv[1:]:
        run_cli()
        return
    App().run()


if __name__ == "__main__":
    main()

# Sokoban Planner

Search and planning project that solves a small Sokoban puzzle with
Breadth-First Search, Uniform-Cost Search, and A* Search.

## Execution

### Install Requirements

```sh
pip install -r requirements.txt # Only install this if you intend on using the GUI
```

### Run the simulator

For the live visualization GUI (at the expense of GUI overhead):

```sh
python main.py
```

For the straight-forward CLI (for purer search algorithm results):

```sh
python main.py --cli
```

## Search result logs

Logs for each algorithm are stored in the ```logs/``` directory.

All subsequent runs (by you) will be added to the this directory appended with ..._#.json.


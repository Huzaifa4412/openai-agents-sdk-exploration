import json
from pathlib import Path
from typing import List
from task_model import Task

FILE = Path("task.json")


def save_tasks(task) -> List[Task]:
    with open(FILE, "w") as f:
        json.dump(task, f, indent=4)


def load_tasks(task: List[Task]):
    with open(FILE, "r") as f:
        return json.load(f)

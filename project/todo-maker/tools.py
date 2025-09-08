from agents import function_tool
from typing import List
from task_model import Task
import storage_manage as storage


@function_tool
def add_task(
    title: str, description: str = None, due_date: str = None, priority: str = "medium"
) -> str:
    """Add a new task to the todo list."""
    tasks = storage.load_tasks()
    task = Task(
        id=len(tasks) + 1,
        title=title,
        description=description,
        due_date=due_date,
        priority=priority,
        completed=False,
    )
    tasks.append(task)
    storage.save_tasks(tasks)
    return f"Task '{task.title}' added."


@function_tool
def list_tasks() -> List[Task]:
    """List all tasks."""
    return storage.load_tasks()


@function_tool
def complete_task(task_id: int) -> str:
    """Mark a task as completed."""
    tasks = storage.load_tasks()
    for t in tasks:
        if t.id == task_id:
            t.completed = True
            return f"Task '{t.title}' marked as completed."
    return f"Task {task_id} not found."

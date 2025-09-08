# models.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class Task(BaseModel):
    id: int
    title: str = Field(..., description="Task title")
    description: Optional[str] = None
    due_date: Optional[datetime] = None
    priority: str = Field(default="medium", description="low, medium, high")
    completed: bool = False

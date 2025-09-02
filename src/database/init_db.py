# Initialize DB once

from __future__ import annotations
from .database import create_tables

def init():
    create_tables()

if __name__ == "__main__":
    init()

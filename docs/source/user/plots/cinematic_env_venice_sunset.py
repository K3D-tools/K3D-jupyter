import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _cinematic_dragon import screenshot


def generate():
    return screenshot('venice_sunset')

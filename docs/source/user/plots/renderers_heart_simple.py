import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _renderers_heart import screenshot


def generate():
    return screenshot('simple')

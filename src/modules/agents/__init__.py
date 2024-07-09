REGISTRY = {}

from .agent import Agent
from .filtered_agent import FilteredAgent
REGISTRY["agent"] = Agent
REGISTRY["filtered_agent"] = FilteredAgent
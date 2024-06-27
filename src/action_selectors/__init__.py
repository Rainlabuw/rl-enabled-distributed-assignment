REGISTRY = {}

from .action_selectors import REDASelector, EpsilonGreedySelector, SoftPoliciesSelector
from .filtered_action_selectors import FilteredREDASelector, FilteredEpsilonGreedySelector, FilteredSoftPoliciesSelector

REGISTRY["reda"] = REDASelector
REGISTRY["epsilon_greedy"] = EpsilonGreedySelector
REGISTRY["soft_policies"] = SoftPoliciesSelector

REGISTRY["filtered_reda"] = FilteredREDASelector
REGISTRY["filtered_epsilon_greedy"] = FilteredEpsilonGreedySelector
REGISTRY["filtered_soft_policies"] = FilteredSoftPoliciesSelector
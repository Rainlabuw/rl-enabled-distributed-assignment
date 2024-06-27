REGISTRY = {}

from .classic_selectors import MultinomialActionSelector, EpsilonGreedyActionSelector, SoftPoliciesSelector
from .sap_selectors import SequentialAssignmentProblemSelector, EpsilonGreedySAPTestActionSelector
from .filtered_sap_selectors import FilteredSAPActionSelector, FilteredEpsGrSAPTestActionSelector
from .filtered_classic_selectors import FilteredEpsilonGreedyActionSelector, FilteredSoftPoliciesSelector

REGISTRY["multinomial"] = MultinomialActionSelector
REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
REGISTRY["soft_policies"] = SoftPoliciesSelector
REGISTRY["sap"] = SequentialAssignmentProblemSelector
REGISTRY["epsilon_greedy_sap_test"] = EpsilonGreedySAPTestActionSelector
REGISTRY["filtered_const_sap"] = FilteredSAPActionSelector
REGISTRY["filtered_const_epsilon_greedy"] = FilteredEpsilonGreedyActionSelector
REGISTRY["filtered_const_epsgr_sap_test"] = FilteredEpsGrSAPTestActionSelector
REGISTRY["filtered_const_soft_policies"] = FilteredSoftPoliciesSelector
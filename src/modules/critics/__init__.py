from .coma import COMACritic
from .filtered_coma import FilteredCOMACritic
from .ac import ACCritic
from .ac_ns import ACCriticNS
REGISTRY = {}

REGISTRY["coma_critic"] = COMACritic
REGISTRY["filtered_coma_critic"] = FilteredCOMACritic
REGISTRY["ac_critic"] = ACCritic
REGISTRY["ac_critic_ns"] = ACCriticNS
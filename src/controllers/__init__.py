REGISTRY = {}

from .basic_controller import BasicMAC
from .jumpstart_controller import JumpstartMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["jumpstart_mac"] = JumpstartMAC

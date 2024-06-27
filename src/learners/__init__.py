from .q_learner import QLearner
from .filtered_q_learner import FilteredQLearner
from .sap_q_learner import SAPQLearner
from .filtered_sap_q_learner import FilteredSAPQLearner
from .coma_learner import COMALearner
from .filtered_coma_learner import FilteredCOMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .actor_critic_pac_learner import PACActorCriticLearner
from .actor_critic_pac_dcg_learner import PACDCGLearner
from .maddpg_learner import MADDPGLearner
from .ppo_learner import PPOLearner
from .filtered_ppo_learner import FilteredPPOLearner
from .cont_ppo_learner import ContinuousPPOLearner
from .bc_learner import BCLearner
from .filtered_bc_learner import FilteredBCLearner
REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["filtered_q_learner"] = FilteredQLearner
REGISTRY["sap_q_learner"] = SAPQLearner
REGISTRY["filtered_sap_q_learner"] = FilteredSAPQLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["filtered_coma_learner"] = FilteredCOMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["filtered_ppo_learner"] = FilteredPPOLearner
REGISTRY["cont_ppo_learner"] = ContinuousPPOLearner
REGISTRY["pac_learner"] = PACActorCriticLearner
REGISTRY["pac_dcg_learner"] = PACDCGLearner
REGISTRY["bc_learner"] = FilteredBCLearner
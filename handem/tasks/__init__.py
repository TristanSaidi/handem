from handem.tasks.ihm_base import IHMBase
from handem.tasks.handem_classify import HANDEM_Classify
from handem.tasks.handem_reconstruct import HANDEM_Reconstruct

task_map = {
    "IHM": IHMBase,
    "HANDEM_Classify": HANDEM_Classify,
    "HANDEM_Reconstruct": HANDEM_Reconstruct,
}
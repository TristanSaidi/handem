from handem.tasks.ihm_base import IHMBase
from handem.tasks.finger_gaiting import FingerGaiting
from handem.tasks.rrt import RRT
from handem.tasks.extract_grasps import ExtractGrasps
from handem.tasks.handem import HANDEM

task_map = {
    "IHM": IHMBase,
    "FingerGaiting": FingerGaiting,
    "RRT": RRT,
    "ExtractGrasps": ExtractGrasps,
    "HANDEM": HANDEM,
}
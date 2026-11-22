"""NEO training that starts from every part of another run's model. Registered as METHODS["neo_warm"].

The lab method loads only the state encoder and decoder from `train.init_from` (the pretrained codec). This one loads all
matching parts, programmer, codebook and executor included, so that training can continue from a model trained in another
setting, for example on one program step. Everything else is the lab method.
"""

from __future__ import annotations

from core.registry import METHODS
from methods.neo import NEOMethod


class NEOWarmStart(NEOMethod):
    init_prefixes = None  # load every key the checkpoint and the model share


METHODS["neo_warm"] = NEOWarmStart

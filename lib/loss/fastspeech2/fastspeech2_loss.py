from lib.loss.combined_loss import CombinedLoss
from .duration_loss import DurationLoss
from .energy_loss import EnergyLoss
from .mel_loss import MelLoss
from .pitch_loss import PitchLoss


class Fastspeech2Loss(CombinedLoss):
    def __init__(self,
                 mel_weight: float = 0.25,
                 pitch_weight: float = 0.25,
                 energy_weight: float = 0.25,
                 duration_weight: float = 0.25):
        losses = {
            'mel spec loss': MelLoss(),
            'pitch loss': PitchLoss(),
            'energy loss': EnergyLoss(),
            'duration loss': DurationLoss(),
        }
        weights = dict(zip(losses.keys(), (mel_weight, pitch_weight, energy_weight, duration_weight)))
        super().__init__(losses, weights)

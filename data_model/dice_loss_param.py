from dataclasses import dataclass


@dataclass
class DiceLossPram:
    smooth: float

    @classmethod
    def from_config(cls, config):
        return cls(smooth=config.smooth)

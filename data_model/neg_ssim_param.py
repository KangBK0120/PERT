from dataclasses import dataclass


@dataclass
class NegSSIMPram:
    window_size: int

    @classmethod
    def from_config(cls, config):
        return cls(window_size=config.window_size)

from dataclasses import dataclass
from typing import List


@dataclass
class GlobalAwareSimParam:
    kernel_sizes: List[int]

    @classmethod
    def from_config(cls, config):
        return cls(kernel_sizes=config.kernel_sizes)

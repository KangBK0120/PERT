from dataclasses import dataclass


@dataclass
class RegionAwareSimilarityParam:
    lambda_out: float
    lambda_p1: float
    lambda_p2: float
    beta_out: float
    beta_p1: float
    beta_p2: float

    @classmethod
    def from_config(cls, config):
        return cls(
            lambda_out=config.lambda_out,
            lambda_p1=config.lambda_p1,
            lambda_p2=config.lambda_p2,
            beta_out=config.beta_out,
            beta_p1=config.beta_p1,
            beta_p2=config.beta_p2,
        )

from abc import ABC, abstractmethod


class Scheduler(ABC):
    @abstractmethod
    def calc_multiplier(self, cur_epoch: int) -> float:
        pass


class MilestoneLR(Scheduler):
    def __init__(self, milestones: list[int], multipliers: list[float]) -> None:
        self.milestones: list[int] = milestones
        self.multipliers: list[float] = multipliers

        if len(self.milestones) == 0 or self.milestones[0] > 0:
            self.milestones = [0] + self.milestones
            self.multipliers = [1] + self.multipliers

        self.pivot: int = 0

    def calc_multiplier(self, cur_epoch: int) -> float:
        while self.pivot + 1 < len(self.milestones) and \
                self.milestones[self.pivot + 1] <= cur_epoch:
            self.pivot += 1
        return self.multipliers[self.pivot]


class ExpDecayLR(Scheduler):
    def __init__(self, init: float, alpha: float) -> None:
        self.init: float = init
        self.alpha: float = alpha

    def calc_multiplier(self, cur_epoch: int) -> float:
        return self.init * self.alpha ** cur_epoch

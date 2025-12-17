from abc import ABC, abstractmethod


class ModelProvider(ABC):
    @abstractmethod
    def generate(self, prompt):
        raise NotImplementedError()

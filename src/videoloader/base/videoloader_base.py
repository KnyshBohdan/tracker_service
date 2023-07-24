from abc import ABC
import numpy as np


class VideoLoaderBase(ABC):
    def get_frame(self) -> np.ndarray:
        pass

    def is_opened(self) -> bool:
        pass

    def get_fps(self) -> int:
        pass

    def open(self, **kwargs):
        pass

    def close(self):
        pass

from abc import ABC


class VideoLoaderBase(ABC):
    def get_frame(self):
        pass

    def open(self, **kwargs):
        pass

    def close(self):
        pass

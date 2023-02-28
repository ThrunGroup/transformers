from .svd import SVDWrapper
from .pca import PCAWrapper
from .vanilla import VanillaWrapper
from utils.constants import SVD, PCA


class AcceleratorFactory:
    @staticmethod
    def get_accelerator(type: str = None):
        if type == SVD:
            return SVDWrapper
        elif type == PCA:
            return PCAWrapper
        elif type is None:
            return VanillaWrapper
        else:
            assert False, "There is no such accelerator."


if __name__ == "__main__":
    accelerator = AcceleratorFactory().get_accelerator("SVD")

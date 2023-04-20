from accelerators.svd import SVDWrapper
from accelerators.pca import PCAWrapper
from accelerators.pruning import PruningWrapper
from accelerators.vanilla import VanillaWrapper
from accelerators.simple_quantization_wrapper import QuantLinearWrapper
from utils.constants import SVD, PCA, QUANTIZATION, PRUNING


class AcceleratorFactory:
    @staticmethod
    def get_accelerator(type: str = None, **kwargs):
        if type == SVD:
            return SVDWrapper
        elif type == PCA:
            return PCAWrapper
        elif type == QUANTIZATION:
            return QuantLinearWrapper
        elif type == PRUNING:
            return PruningWrapper
        elif type is None:
            return VanillaWrapper
        else:
            assert False, "There is no such accelerator."


if __name__ == "__main__":
    accelerator = AcceleratorFactory().get_accelerator("SVD")

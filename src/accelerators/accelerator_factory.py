from .svd import SVDWrapper


class AcceleratorFactory:
    """

    """
    @staticmethod
    def get_accelerator(type: str):
        if type == "SVD":
            return SVDWrapper
        else:
            assert False, "There is no such accelerator."


if __name__ == "__main__":
    accelerator = AcceleratorFactory().get_accelerator("SVD")
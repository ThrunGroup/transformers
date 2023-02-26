from svd import SVDWrapper


class AcceleratorFactory:
    @staticmethod
    def get_accelerator(type: str, **kwargs):
        if type == "SVD":
            return SVDWrapper(**kwargs)
        else:
            assert False, "There is no such accelerator."


if __name__ == "__main__":
    factory = AcceleratorFactory()
    factory.get_accelerator("test", k=10)

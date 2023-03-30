from utils.constants import (
    # Accelerators
    SVD,

    # Models
    GPT2,
    GPT2_MEDIUM,

    # Parameters
    NUM_BLOCKS_GPT2,

    # Datasets
    BILLSUM,
    OPENWEBTEXT,
    SQUAD,
)
from create_models import create_model

if __name__ == '__main__':
    create_model(model_type=GPT2_MEDIUM,
                 dataset_name=BILLSUM,
                 num_epochs=5,
                 layers_to_freeze=f"0-{NUM_BLOCKS_GPT2 - 1}",
                 layers_to_accelerate=f"{NUM_BLOCKS_GPT2 - 1}",
                 train_accelerated_layers=True,
                 accelerator_type=SVD,
                 k=64)

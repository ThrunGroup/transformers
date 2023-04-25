from vizier.service import clients
from vizier.service import pyvizier as vz

from create_models import create_model
from utils.constants import (
    # Accelerators
    SVD,

    # Models
    GPT2,
    GPT2_MEDIUM,

    # Parameters
    NUM_BLOCKS_GPT2,
    NUM_BLOCKS_GPT2_MEDIUM,

    # Datasets
    BILLSUM,
    OPENWEBTEXT,
    SQUAD,
)


def run_vizier_experiment(study, model_type: str, dataset_name: str, num_experiments: int = 10):
    # Run the experiments multiple times
    for i in range(num_experiments):
        suggestions = study.suggest(count=1)
        for suggestion in suggestions:
            params = suggestion.parameters
            k, first_block_to_accelerate, last_block_to_accelerate, num_blocks_to_finetune = list(map(int,
                                                                                                      params.values()))
            blocks_to_accelerate = f"{first_block_to_accelerate}-{last_block_to_accelerate}"
            blocks_to_freeze = f"0-{NUM_BLOCKS_GPT2 - num_blocks_to_finetune}"

            evaluation_logs = create_model(model_type=model_type,
                                           dataset_name=dataset_name,
                                           num_epochs=1,
                                           layers_to_freeze=blocks_to_freeze,
                                           layers_to_accelerate=blocks_to_accelerate,
                                           train_accelerated_layers=True,
                                           accelerator_type=SVD,
                                           k=k)
            rouge = evaluation_logs['eval_rougeLsum']
            inference_time = evaluation_logs['inference_time']
            suggestion.complete(vz.Measurement({'rouge': rouge,
                                                'inference_time': inference_time}))
            print(f'Iteration {i}, suggestion {params} led to rouge score {rouge} and inference_time {inference_time}.')

    # Get the optimal results
    for optimal_trial in study.optimal_trials():
        optimal_trial = optimal_trial.materialize()
        print("Optimal Trial Suggestion and Objective:", optimal_trial.parameters,
              optimal_trial.final_measurement)


if __name__ == '__main__':
    # Algorithm, search space, and metrics.
    study_config = vz.StudyConfig(algorithm='NSGA2')
    study_config.search_space.root.add_discrete_param('k', [32, 64, 128, 256])
    study_config.search_space.root.add_int_param('first_block_to_accelerate', 0, NUM_BLOCKS_GPT2 // 2)
    study_config.search_space.root.add_int_param('last_block_to_accelerate', NUM_BLOCKS_GPT2 // 2, NUM_BLOCKS_GPT2)
    study_config.search_space.root.add_int_param('num_blocks_to_finetune', 1, 4)
    study_config.search_space.root.add_categorical_param('accelerator', [SVD, QUNA])
    study_config.metric_information.append(vz.MetricInformation('rouge', goal=vz.ObjectiveMetricGoal.MAXIMIZE))
    study_config.metric_information.append(vz.MetricInformation('inference_time', goal=vz.ObjectiveMetricGoal.MINIMIZE))

    """
    quantization for different layers (attention / mlp) -> these can be parameters
    maybe can do a few training steps
    quantization > svd? --> get more trend, intuition 
    """

    # Setup client and begin optimization. Vizier Service will be implicitly created.
    study = clients.Study.from_study_config(study_config, owner='lab', study_id='fast_transformer')
    run_vizier_experiment(study, model_type=GPT2, dataset_name=BILLSUM, num_experiments=10)

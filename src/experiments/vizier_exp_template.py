from vizier.service import clients
from vizier.service import pyvizier as vz


# Objective function to maximize.
def evaluate_max(w: float, x: int, y: float, z: str) -> float:
    return w ** 2 - y ** 2 + x * ord(z)


def evaluate_min(w: float, x: int, y: float, z: str) -> float:
    return -w ** 2 - x * ord(z)


# Algorithm, search space, and metrics.
study_config = vz.StudyConfig(algorithm='NSGA2')
study_config.search_space.root.add_float_param('w', 0.0, 5.0)
study_config.search_space.root.add_int_param('x', -2, 2)
study_config.search_space.root.add_discrete_param('y', [0.3, 7.2])
study_config.search_space.root.add_categorical_param('z', ['a', 'g', 'k'])
study_config.metric_information.append(vz.MetricInformation('metric_name1', goal=vz.ObjectiveMetricGoal.MAXIMIZE))
study_config.metric_information.append(vz.MetricInformation('metric_name2', goal=vz.ObjectiveMetricGoal.MINIMIZE))

# Setup client and begin optimization. Vizier Service will be implicitly created.
study = clients.Study.from_study_config(study_config, owner='my_name', study_id='example')
for i in range(3):
    suggestions = study.suggest(count=1)
    for suggestion in suggestions:
        params = suggestion.parameters
        max_objective = evaluate_max(params['w'], params['x'], params['y'], params['z'])
        min_objective = evaluate_min(params['w'], params['x'], params['y'], params['z'])
        print(f'Iteration {i}, suggestion {params} led to objective value {max_objective}, {min_objective}.')
        final_measurement = vz.Measurement({'metric_name1': max_objective, 'metric_name2': min_objective})
        suggestion.complete(final_measurement)

for optimal_trial in study.optimal_trials():
    optimal_trial = optimal_trial.materialize()
    print("Optimal Trial Suggestion and Objective:", optimal_trial.parameters,
          optimal_trial.final_measurement)

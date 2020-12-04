from secondary_structure import SecondaryStructureTask
from task import Task
from freeze_weights import FreezeWeights
from tensorflow.keras import Model, Sequential
from typing import Dict, List, Type


class TaskBuilder:

    tasks: Dict[str, Type[Task]] = {
        'secondary_structure': SecondaryStructureTask,
    }

    def build_tasks(self, task_names: List[str]) -> List[Task]:
        return [TaskBuilder.build_task(task_name) for task_name in task_names]

    def build_task_model(self,
                         embedding_model: Model,
                         tasks: List[Task],
                         freeze_embedding_weights: bool) -> Model:
        layers = [embedding_model]

        if freeze_embedding_weights:
            layers.append(FreezeWeights())

        for task in tasks:
            layers = task.build_output_model(layers)

        return Sequential(layers)

    def add_task(self, task_name: str, task: Type[Task]) -> None:
        assert isinstance(task, type)
        self.tasks[task_name] = task

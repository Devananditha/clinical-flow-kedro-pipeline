"""Data Engineering Pipeline definition for Kedro."""

import importlib

try:
    _kedro_pipe = importlib.import_module("kedro.pipeline")
    Pipeline = getattr(_kedro_pipe, "Pipeline")
    node = getattr(_kedro_pipe, "node")
    pipeline = getattr(_kedro_pipe, "pipeline")
except (ImportError, ModuleNotFoundError):
    # Graceful fallback shim when running in an environment prior to installing requirements.txt
    class Node:
        def __init__(self, func, inputs, outputs, name=None, tags=None):
            self.func = func
            self.inputs = inputs
            self.outputs = outputs
            self.name = name
            self.tags = tags or []

    class Pipeline:
        def __init__(self, nodes):
            self.nodes = nodes

    def node(func, inputs, outputs, name=None, tags=None):
        return Node(func=func, inputs=inputs, outputs=outputs, name=name, tags=tags)

    def pipeline(nodes):
        return Pipeline(nodes=nodes)

from .nodes import (
    validate_and_ingest_bronze_to_silver,
    build_patient_flow_trajectories,
    simulate_department_surge_capacity,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Instantiate the Clinical Flow Medallion Data Engineering Pipeline (Phase 2).

    Returns:
        Kedro Pipeline chaining:
          1. Bronze -> Silver (HIPAA Safe Harbor De-identification & Cleansing)
          2. Silver -> Gold (Windowed Patient Flow & Ward Trajectory Modeling)
          3. Gold -> Feature (Discrete-Event Surge Simulation & Power BI Export)
    """
    return pipeline(
        [
            node(
                func=validate_and_ingest_bronze_to_silver,
                inputs=["admissions", "transfers", "services"],
                outputs=["int_admissions", "int_transfers", "int_services"],
                name="validate_and_ingest_bronze_to_silver_node",
                tags=["bronze_to_silver", "hipaa_deidentification", "cleansing"],
            ),
            node(
                func=build_patient_flow_trajectories,
                inputs=["int_admissions", "int_transfers", "int_services"],
                outputs="prm_patient_flow",
                name="build_patient_flow_trajectories_node",
                tags=["silver_to_gold", "window_trajectories", "icu_features"],
            ),
            node(
                func=simulate_department_surge_capacity,
                inputs=[
                    "prm_patient_flow",
                    "params:surge_multiplier",
                    "params:bed_turnover_lead_hours",
                    "params:standard_target_occupancy",
                ],
                outputs="feat_bed_surge_metrics",
                name="simulate_department_surge_capacity_node",
                tags=["gold_to_feature", "discrete_event_surge", "powerbi_mart"],
            ),
        ]
    )

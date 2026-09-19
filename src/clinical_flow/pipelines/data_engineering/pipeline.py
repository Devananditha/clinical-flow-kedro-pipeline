"""Data Engineering Pipeline definition for Kedro."""

try:
    from kedro.pipeline import Pipeline, node, pipeline
except ImportError:
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
    clean_admissions,
    clean_transfers,
    clean_services,
    create_patient_flow,
    compute_bed_surge_metrics,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Instantiate the Clinical Flow Medallion Data Engineering Pipeline.

    Returns:
        Kedro Pipeline composing Bronze->Silver, Silver->Gold, and Gold->Feature transformations.
    """
    return pipeline(
        [
            node(
                func=clean_admissions,
                inputs="admissions",
                outputs="int_admissions",
                name="clean_admissions_node",
                tags=["silver", "admissions"],
            ),
            node(
                func=clean_transfers,
                inputs="transfers",
                outputs="int_transfers",
                name="clean_transfers_node",
                tags=["silver", "transfers"],
            ),
            node(
                func=clean_services,
                inputs="services",
                outputs="int_services",
                name="clean_services_node",
                tags=["silver", "services"],
            ),
            node(
                func=create_patient_flow,
                inputs=["int_admissions", "int_transfers", "int_services"],
                outputs="prm_patient_flow",
                name="create_patient_flow_node",
                tags=["gold", "patient_flow"],
            ),
            node(
                func=compute_bed_surge_metrics,
                inputs=[
                    "prm_patient_flow",
                    "params:surge_multiplier",
                    "params:bed_turnover_lead_hours",
                ],
                outputs="feat_bed_surge_metrics",
                name="compute_bed_surge_metrics_node",
                tags=["feature", "surge_capacity"],
            ),
        ]
    )

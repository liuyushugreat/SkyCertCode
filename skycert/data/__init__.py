from .synthetic import UAMDataset, make_uam_dataset
from .threats import (
    adaptive_feature_attack,
    apply_threat,
    corrupt_rules,
    corrupt_rules_targeted,
    inject_rule_noise,
    perturb_features,
    poison_calibration_labels,
    rank_rules_by_critical_impact,
    shift_covariates,
)
from .trajectory_sim import TrajectoryCaseStudy, make_trajectory_case_study

__all__ = [
    "UAMDataset",
    "make_uam_dataset",
    "TrajectoryCaseStudy",
    "make_trajectory_case_study",
    "adaptive_feature_attack",
    "apply_threat",
    "corrupt_rules",
    "corrupt_rules_targeted",
    "inject_rule_noise",
    "perturb_features",
    "poison_calibration_labels",
    "rank_rules_by_critical_impact",
    "shift_covariates",
]

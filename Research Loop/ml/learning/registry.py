# ml/learning/registry.py

from ml.learning.classical.sklearn_erm import SklearnERM
from ml.learning.classical.neural_erm import NeuralERM

from ml.learning.domain_generalization.irm import IRM
from ml.learning.domain_generalization.groupdro import GroupDRO
from ml.learning.domain_generalization.vrex import VREx
from ml.learning.domain_generalization.coral import CORAL
from ml.learning.domain_generalization.mmd import MMD

from ml.learning.domain_adaptation_unlabeled.deep_coral import DeepCORAL
from ml.learning.domain_adaptation_unlabeled.dann import DANN
from ml.learning.domain_adaptation_unlabeled.mcd import MCD
from ml.learning.domain_adaptation_unlabeled.importance_weighting import (
    ImportanceWeighting,
)

from ml.learning.domain_adaptation_labeled.joint_supervised import (
    JointSupervised,
)
from ml.learning.domain_adaptation_labeled.supervised_dann import (
    SupervisedDANN,
)

from ml.learning.source_free.source_free_unlabeled.shot import SHOT
from ml.learning.source_free.source_free_unlabeled.nrc import NRC
from ml.learning.source_free.source_free_unlabeled.sfda_de import SFDADE

from ml.learning.source_free.source_free_labeled.linear_probe import (
    LinearProbe,
)
from ml.learning.source_free.source_free_labeled.fine_tuning import (
    FineTuning,
)
from ml.learning.source_free.source_free_labeled.lp_ft import LPFT
from ml.learning.source_free.source_free_labeled.l2_sp import L2SP

# ============================================================
# Main references
# ============================================================

# Classical
# ERM:
#   Standard empirical risk minimization baseline.
#   No single method-specific reference required.

# Domain Generalization
# IRM:
#   Arjovsky et al. (2019)
#   "Invariant Risk Minimization"
#
# GroupDRO:
#   Sagawa et al. (ICLR 2020)
#   "Distributionally Robust Neural Networks for Group Shifts:
#   On the Importance of Regularization for Worst-Case Generalization"
#
# VREx:
#   Krueger et al. (ICML 2021)
#   "Out-of-Distribution Generalization via Risk Extrapolation (REx)"
#
# CORAL:
#   Sun and Saenko (ECCV 2016)
#   "Deep CORAL: Correlation Alignment for Deep Domain Adaptation"
#   Adapted here to align multiple source domains.
#
# MMD:
#   Gretton et al. (JMLR 2012)
#   "A Kernel Two-Sample Test"
#   MMD is used here as the source-domain alignment penalty.

# Unlabeled Domain Adaptation
# Deep CORAL:
#   Sun and Saenko (ECCV 2016)
#   "Deep CORAL: Correlation Alignment for Deep Domain Adaptation"
#
# DANN:
#   Ganin et al. (JMLR 2016)
#   "Domain-Adversarial Training of Neural Networks"
#
# MCD:
#   Saito et al. (CVPR 2018)
#   "Maximum Classifier Discrepancy for Unsupervised Domain Adaptation"
#
# Importance Weighting / KLIEP:
#   Sugiyama et al. (NeurIPS 2007)
#   "Direct Importance Estimation with Model Selection and Its
#   Application to Covariate Shift Adaptation"
#
# Importance Weighting / uLSIF:
#   Kanamori et al. (JMLR 2009)
#   "A Least-squares Approach to Direct Importance Estimation"

# Labeled Domain Adaptation
# Joint Supervised:
#   Standard labeled source + target pooling baseline.
#   No single method-specific reference required.
#
# Supervised DANN:
#   Supervised extension of:
#   Ganin et al. (JMLR 2016)
#   "Domain-Adversarial Training of Neural Networks"

# Source-Free Unlabeled
# SHOT:
#   Liang et al. (ICML 2020)
#   "Do We Really Need to Access the Source Data?
#   Source Hypothesis Transfer for Unsupervised Domain Adaptation"
#
# NRC:
#   Yang et al. (NeurIPS 2021)
#   "Exploiting the Intrinsic Neighborhood Structure
#   for Source-free Domain Adaptation"
#
# SFDA-DE:
#   Ding et al. (CVPR 2022)
#   "Source-Free Domain Adaptation via Distribution Estimation"

# Source-Free Labeled
# Linear Probe:
#   Standard transfer-learning baseline.
#
# Fine-Tuning:
#   Standard transfer-learning baseline.
#
# LP-FT:
#   Kumar et al. (ICLR 2022)
#   "Fine-Tuning Can Distort Pretrained Features
#   and Underperform Out-of-Distribution"
#
# L2-SP:
#   Li et al. (ICML 2018)
#   "Explicit Inductive Bias for Transfer Learning
#   with Convolutional Networks"


LEARNING_ALGORITHMS = {
    # Classical
    "sklearn_erm": SklearnERM,
    "neural_erm": NeuralERM,

    # Domain generalization
    "irm": IRM,
    "groupdro": GroupDRO,
    "vrex": VREx,
    "coral": CORAL,
    "mmd": MMD,

    # Unlabeled domain adaptation
    "deep_coral": DeepCORAL,
    "dann": DANN,
    "mcd": MCD,
    "importance_weighting": ImportanceWeighting,

    # Labeled domain adaptation
    "joint_supervised": JointSupervised,
    "supervised_dann": SupervisedDANN,

    # Source-free unlabeled
    "shot": SHOT,
    "nrc": NRC,
    "sfda_de": SFDADE,

    # Source-free labeled
    "linear_probe": LinearProbe,
    "fine_tuning": FineTuning,
    "lp_ft": LPFT,
    "l2_sp": L2SP,
}


def get_learning_algorithm(name, params=None):
    if name not in LEARNING_ALGORITHMS:
        raise ValueError(
            f"Unknown learning algorithm '{name}'. "
            f"Available: {sorted(LEARNING_ALGORITHMS)}"
        )

    params = params or {}

    return LEARNING_ALGORITHMS[name](**params)
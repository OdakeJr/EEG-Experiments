from ml.domain_discrepancy.mmd import (
    compute_mmd,
)

from ml.domain_discrepancy.energy import (
    compute_energy,
)


DOMAIN_DISCREPANCIES = {
    "mmd": compute_mmd,
    "energy": compute_energy,
}


def get_domain_discrepancy(name):
    if name not in DOMAIN_DISCREPANCIES:
        raise ValueError(
            f"Unknown domain discrepancy: {name}. "
            f"Available: {list(DOMAIN_DISCREPANCIES)}"
        )

    return DOMAIN_DISCREPANCIES[name]
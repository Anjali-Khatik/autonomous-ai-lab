"""[V2] compute_impact() — arithmetic on USER-SUPPLIED business_params only.

TODO: implement for V2. Must NEVER fabricate ARPU/revenue/ROI — only compute
from business_params fields the user explicitly supplied. If business_params
is None, the caller (Business Analyst) should treat impact as null and skip
calling this function entirely.
"""


def compute_impact(metrics: dict, business_params: dict) -> dict:
    raise NotImplementedError("[V2] compute_impact not implemented in V1 — see docstring")

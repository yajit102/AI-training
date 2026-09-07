"""FlowPredict TAT prediction pipeline.

Strengthened successor to the standalone "Master_geminicode" prototype
script. Adds hierarchical entities (case / issuing authority / country /
check type), disruption-signal decay features, right-censoring-aware
(survival) quantile training, Mondrian conformal calibration, and
copula-based critical-path aggregation across multi-check dossiers.
"""

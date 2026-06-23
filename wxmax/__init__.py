"""wxmax — calibrated multi-model ensemble for daily-max temperature (US).

Package layout:
  ingest/    data acquisition (ground truth + forecasts)
  features/  feature store construction
  models/    MOS blend + probabilistic head
  calibrate/ conformal prediction intervals
  verify/    metrics + back-test harness
  serve/     live API (P7)
"""

__version__ = "0.1.0"

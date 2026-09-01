## Project layout

```text
iins/travel_time.py       Grid speed and cross-grid travel-time estimation
iins/global_planner.py    Occupancy-aware global planning with CBF constraints
iins/local_refinement.py  Longitudinal/lateral spatial-domain MPC refinement
```

The MPC implementation includes backward computation of the feedback law,
forward state rollout, steering saturation, steering-rate limits and
longitudinal smoothness limits.

## Installation and quick start

```bash
python -m pip install -e .
```

Minimal use:

```python
import numpy as np
from iins import (
    GlobalPlannerConfig,
    GridSpec,
    GridTravelTimeEstimator,
    SpatialGlobalPlanner,
    SpatialMPCRefiner,
)

grid = GridSpec(0, 100, -6, 6, ds=2, dy=1)
# Every historical trajectory has columns [time, longitudinal_s, lateral_y].
history = [np.array([[0, 0, 0], [1, 10, 0], [2, 20, 0]], dtype=float)]
travel_time = GridTravelTimeEstimator(grid).fit(history)

global_path = SpatialGlobalPlanner(
    travel_time, GlobalPlannerConfig(horizon=60, ds=1.0)
).plan()
local_path = SpatialMPCRefiner().refine(global_path, desired_speed=12.0)
```

Dynamic obstacles can be supplied to the global planner as
`ObstacleProfile` objects. Each profile provides a predicted lateral center as
a function of longitudinal station and a squared safety-distance threshold.

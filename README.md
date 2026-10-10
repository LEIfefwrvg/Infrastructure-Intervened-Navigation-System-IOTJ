# Infrastructure-Intervened Navigation System



## 1. `travel_time.py` 的使用

`iins/travel_time.py` 将历史占用网格和车辆轨迹转换为未来速度场，并提供规划器使用的通行时间查询。

| 输入 | 形状 | 含义 |
| --- | --- | --- |
| `occupancy` | `(H, W, 7)` | 连续七帧历史网格占用，历史跨度 3 秒、帧间隔 0.5 秒 |
| `trajectories` | `(N, 7, 3)` | 每辆车的历史 `[world_x, world_y, speed]` |
| `agent_mask` | `(N,)` | 有效车辆标记 |
| `epoch` | 标量 | 当前时刻，用作车辆到达时间的时间基准 |

```python
from iins.travel_time import GridSpec, GridTravelTimeEstimator
grid = GridSpec.from_road(road, start_s=...)
estimator = GridTravelTimeEstimator(grid)
forecast = estimator.predict(occupancy, trajectories, agent_mask=np.array([True]), epoch=0.)
field = forecast.field(future_index=0)
```
弯曲道路需要将世界坐标预测转换到规划器的道路坐标：使用 `RoadTravelTimeField(forecast, road, start_s=...)`

## 2. Global path planning 的使用

`iins/global_planner.py` 中的 `SpatialGlobalPlanner` 根据通行时间场，在空间域优化转向角和路径，同时处理横向边界与障碍物约束。

```python
from iins.global_planner import GlobalPlannerConfig, SpatialGlobalPlanner

planner = SpatialGlobalPlanner(
    field,
    GlobalPlannerConfig(
        horizon=60,          # 基础空间网格的区间数
        ds=1.0,              # 基础节点间距，单位米
        control_step=3.0,    # 相邻区间约每 3 米共用一个控制变量
        wheelbase=2.6,       # 轴距，单位米
        variable_steps=False,
    ),
)
global_path = planner.plan(
    start_s=0.,
    initial_y=0.,
    initial_heading=0.,
    curvature=0.,
    obstacles=(),
)
```

障碍物使用 `ObstacleProfile(s, y, safe_squared_distance=...)` 描述：`s` 与 `y` 是等长数组，给出有效纵向范围内的障碍物横向位置，`safe_squared_distance` 是要求的横向间距平方，单位 m²。

## 3. Local path refinement 的使用

`iins/local_refinement.py` 中的 `SpatialMPCRefiner` 在全局路径基础上联合优化到达时间、横向位置、速度和航向，处理速度、加减速度、转向及车辆到达时间间隔等约束。

```python
from iins.local_refinement import LocalRefinementConfig, SpatialMPCRefiner

refiner = SpatialMPCRefiner(
    LocalRefinementConfig(
        horizon=25,
        spatial_step=1.0,
        wheelbase=2.6,
        speed_limit=20.0,
        max_acceleration=2.0,
        max_deceleration=6.0,
    ),
)
local_path = refiner.refine(
    global_path,
    desired_speed=8.0,       # 可以是标量，也可以是与实际局部节点数一致的数组
    initial_state=np.array([0., global_path.y[0], 1. / 8., global_path.heading[0]]),
    initial_steering=0.,
    initial_moderation=0.,
    road_bounds=(-3.5, 3.5),
    vehicle_arrivals=(),
    obstacles=(),
)
```

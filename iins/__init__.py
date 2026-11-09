"""IINS public API, loaded lazily so module CLIs run without import side effects."""
from importlib import import_module

_MODULES = {
    'GlobalPath':'global_planner','GlobalPlannerConfig':'global_planner',
    'ObstacleProfile':'global_planner','SpatialGlobalPlanner':'global_planner',
    'LocalPath':'local_refinement','LocalRefinementConfig':'local_refinement','SpatialMPCRefiner':'local_refinement',
    'GridSpec':'travel_time','GridTravelTimeEstimator':'travel_time','TravelTimeField':'travel_time',
    'FlowForecast':'travel_time','VehicleArrival':'travel_time',
    'RoadTravelTimeField':'travel_time','estimate_ego_arrival_times':'travel_time',
}
__all__=list(_MODULES)

def __getattr__(name):
    if name not in _MODULES: raise AttributeError(name)
    value=getattr(import_module('.'+_MODULES[name],__name__),name)
    globals()[name]=value
    return value

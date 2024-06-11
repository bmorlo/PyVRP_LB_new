import matplotlib.pyplot as plt
from tabulate import tabulate
from vrplib import read_solution

from pyvrp import Model, read
from pyvrp.plotting import (
    plot_coordinates,
    plot_instance,
    plot_result,
    plot_route_schedule,
)
from pyvrp.stop import MaxIterations, MaxRuntime

INSTANCE = read("BM_dev/BM_instances/X-n214-k11-C20_unit-demand.vrp", round_func="round")

model = Model.from_data(INSTANCE)
seeds = [42, 12, 37, 6, 24, 68, 153, 402, 87, 2]
for seed in seeds:
    result = model.solve(stop=MaxRuntime(102), seed=seed, display=True)
    print(result)
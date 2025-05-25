import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n = 100  # Number of components
T = 20  # Time horizon
M = 1000  # Big-M constant for constraints
K = 2   # Number of maintenance crews

# Component-specific parameters
RUL = [np.random.randint(50, 100) for _ in range(n)]  # Predicted RUL for each component
c_pm = [10 for _ in range(n)]  # Preventive maintenance cost
c_cm = [50 for _ in range(n)]  # Corrective maintenance cost (higher due to failure)
t_pm = [5 for _ in range(n)]   # Preventive maintenance time
t_cm = [10 for _ in range(n)]  # Corrective maintenance time

# Initialize Gurobi model
model = gp.Model("PredictiveMaintenance")

# Decision variables
x = model.addVars(n, T, vtype=GRB.BINARY, name="x")  # x[i,t]: maintenance of component i starts at time t
y = model.addVars(n, vtype=GRB.BINARY, name="y")     # y[i]: 1 if component i fails (CM), 0 if PM

# Completion time variable (for total maintenance completion time)
C_max = model.addVar(vtype=GRB.CONTINUOUS, name="C_max")

# Objective 1: Minimize total maintenance cost
cost = gp.quicksum(c_pm[i] * (1 - y[i]) + c_cm[i] * y[i] for i in range(n))

# Objective 2: Minimize maximum completion time (C_max)
# C_max is constrained to be greater than the completion time of each maintenance task
for i in range(n):
    for t in range(T):
        model.addConstr(C_max >= (t + t_pm[i] * (1 - y[i]) + t_cm[i] * y[i]) * x[i, t])

# Constraints
# 1. Each component is maintained exactly once
for i in range(n):
    model.addConstr(gp.quicksum(x[i, t] for t in range(T)) == 1, name=f"OneMaintenance_{i}")

# 2. Failure condition: y[i] = 1 if maintenance starts after RUL
for i in range(n):
    for t in range(T):
        if t > RUL[i]:
            model.addConstr(x[i, t] <= y[i], name=f"Failure_{i}_{t}")

# 3. Resource constraint: At most K crews can perform maintenance at any time
for t in range(T):
    model.addConstr(
        gp.quicksum(x[i, t_prime] for i in range(n) for t_prime in range(max(0, t - t_cm[i] + 1), min(T, t + t_pm[i]))) <= K,
        name=f"Resource_{t}"
    )

# ε-constraint method: Generate Pareto solutions by varying ε for completion time
epsilon_values = [20, 40, 60, 80, 100]  # Possible bounds for C_max
pareto_solutions = []

for epsilon in epsilon_values:
    # Reset model
    model.reset()
    
    # Add ε-constraint for C_max
    model.addConstr(C_max <= epsilon, name="Epsilon_Constraint")
    
    # Set objective to minimize cost
    model.setObjective(cost, GRB.MINIMIZE)
    
    # Optimize
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        solution = {
            'epsilon': epsilon,
            'cost': model.objVal,
            'C_max': C_max.X,
            'maintenance_schedule': [(i, t, x[i, t].X, y[i].X) for i in range(n) for t in range(T) if x[i, t].X > 0.5]
        }
        pareto_solutions.append(solution)

# Print Pareto solutions
print("\nPareto-Optimal Solutions:")
for sol in pareto_solutions:
    print(f"\nε = {sol['epsilon']}:")
    print(f"  Total Cost: {sol['cost']}")
    print(f"  Max Completion Time: {sol['C_max']}")
    print("  Maintenance Schedule:")
    for i, t, x_val, y_val in sol['maintenance_schedule']:
        maintenance_type = "CM" if y_val > 0.5 else "PM"
        print(f"    Component {i} at time {t}: {maintenance_type}")
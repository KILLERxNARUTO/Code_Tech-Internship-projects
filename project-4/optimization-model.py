# Import necessary libraries
import pandas as pd
import seaborn as sns
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import matplotlib.pyplot as plt

# ---------- Data Definitions ----------

# Warehouses and their available supply units
warehouses = ['Mumbai', 'Delhi', 'Chennai']
supply = {
    'Mumbai': 250,
    'Delhi': 300,
    'Chennai': 400
}

# Stores and their demand requirements
stores = ['Bangalore', 'Hyderabad', 'Jaipur', 'Kolkata']
demand = {
    'Bangalore': 200,
    'Hyderabad': 225,
    'Jaipur': 275,
    'Kolkata': 250
}

# Transportation cost per unit from each warehouse to each store (in ₹)
costs = {
    ('Mumbai', 'Bangalore'): 8,
    ('Mumbai', 'Hyderabad'): 6,
    ('Mumbai', 'Jaipur'): 10,
    ('Mumbai', 'Kolkata'): 9,
    ('Delhi', 'Bangalore'): 9,
    ('Delhi', 'Hyderabad'): 12,
    ('Delhi', 'Jaipur'): 13,
    ('Delhi', 'Kolkata'): 7,
    ('Chennai', 'Bangalore'): 14,
    ('Chennai', 'Hyderabad'): 9,
    ('Chennai', 'Jaipur'): 16,
    ('Chennai', 'Kolkata'): 5
}

# ---------- Linear Programming Model ----------

# Initialize LP model to minimize total transportation cost
model = LpProblem("Logistics_Cost_Minimization", LpMinimize)

# Define decision variables for each route (shipment quantity from warehouse to store)
x = LpVariable.dicts("Route", (warehouses, stores), lowBound=0, cat='Continuous')

# Define the objective function: total transportation cost
model += lpSum(costs[(w, s)] * x[w][s] for w in warehouses for s in stores), "Total Transportation Cost"

# Add supply constraints: don't ship more than the warehouse capacity
for w in warehouses:
    model += lpSum(x[w][s] for s in stores) <= supply[w], f"Supply_Constraint_{w}"

# Add demand constraints: ensure each store's demand is met or exceeded
for s in stores:
    model += lpSum(x[w][s] for w in warehouses) >= demand[s], f"Demand_Constraint_{s}"

# ---------- Solve the Model ----------
model.solve()

# Print the solution status (Optimal, Infeasible, etc.)
print(f"Solution Status: {LpStatus[model.status]}")

# ---------- Collect and Display Results ----------

# Store results in a list
results = []
for w in warehouses:
    for s in stores:
        quantity = x[w][s].varValue  # Quantity shipped from w to s
        if quantity > 0:
            cost = costs[(w, s)]
            total = quantity * cost
            results.append([w, s, quantity, cost, total])

# Convert results to DataFrame for better readability
df = pd.DataFrame(results, columns=['From', 'To', 'Units', 'Unit Cost', 'Total Cost'])

# Display the optimal shipping plan
print(df)

# Display the total minimum cost
print(f"\n✅ Total Minimum Transportation Cost: ₹{value(model.objective):,.2f}")

# ---------- Visualization ----------

# Plot a bar chart to show how units are distributed from warehouses to stores
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='To', y='Units', hue='From')
plt.title("Shipment Distribution from Warehouses to Stores")
plt.ylabel("Units Shipped")
plt.xlabel("Stores")
plt.legend(title="Warehouse")
plt.tight_layout()
plt.show()


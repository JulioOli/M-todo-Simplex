Simplex Solver in Python

This project implements the Simplex method for solving linear programming problems. The Simplex method is used to maximize or minimize a linear objective function subject to linear equality and inequality constraints.

### Features:
- Interactive user input for defining the problem.
- Step-by-step explanation of the Simplex tableau.
- Handles slack variables and performs pivoting operations.
- Verifies the solution for feasibility and optimality.

### How to Use:
1. Run the program and choose an option from the menu:
   - Solve a new problem by providing the coefficients of the constraints and the objective function.
   - Run a predefined example.
2. Follow the prompts to input the problem data.
3. View the solution and the optimal value of the objective function.

### Example Problem:
Maximize Z = 3x₁ + 2x₂  
Subject to:
- 2x₁ + x₂ ≤ 20
- x₁ + 2x₂ ≤ 20
- x₁ ≤ 10

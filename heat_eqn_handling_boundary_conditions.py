import numpy as np
import scipy.linalg as lin
import matplotlib.pyplot as plt

class TwoDHeatEquationSolver:
    def __init__(self, Nx, Ny, case=1, alpha=0.01):
        """
        Initialize the 2D heat equation solver
        
        Parameters:
        - Nx: Number of grid points in x direction
        - Ny: Number of grid points in y direction
        - case: Problem case (1-4)
        - alpha: Thermal diffusivity (only used in time-dependent case)
        """
        self.Nx = Nx
        self.Ny = Ny
        self.case = case
        self.alpha = alpha
        
        # Grid spacing
        self.dx = 1.0 / (Nx - 1)
        self.dy = 1.0 / (Ny - 1)
        
    def get_source_term(self, x, y):
        """
        Generate source term based on the case
        """
        if self.case == 1:
            return 8 * np.pi**2 * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
        return 0
    
    def apply_boundary_conditions(self, T):
        """
        Apply boundary conditions based on the case
        """
        if self.case == 2:
            # Bottom boundary condition
            T[0, :] = np.sin(np.pi * np.linspace(0, 1, self.Nx))
        elif self.case == 3:
            # Left boundary condition
            T[:, 0] = np.linspace(0, 1, self.Ny) * (1 - np.linspace(0, 1, self.Ny))
        
        return T
    
    def build_coefficient_matrix(self):
        """
        Construct the coefficient matrix for the linear system
        """
        N = (self.Nx - 2) * (self.Ny - 2)
        A = np.zeros((N, N))
        
        # Coefficient for central difference approximation
        coeff = 1 / (self.dx**2 + self.dy**2)
        
        for i in range(self.Nx - 2):
            for j in range(self.Ny - 2):
                row = j * (self.Nx - 2) + i
                
                # Central node
                A[row, row] = -4 * coeff
                
                # Neighboring nodes
                if i > 0:
                    A[row, row - 1] = coeff  # Left
                if i < self.Nx - 3:
                    A[row, row + 1] = coeff  # Right
                if j > 0:
                    A[row, row - (self.Nx - 2)] = coeff  # Bottom
                if j < self.Ny - 3:
                    A[row, row + (self.Nx - 2)] = coeff  # Top
        
        return A
    
    def build_rhs_vector(self):
        """
        Construct the right-hand side vector
        """
        b = np.zeros((self.Nx - 2) * (self.Ny - 2))
        
        # Add source terms for non-homogeneous cases
        for i in range(self.Nx - 2):
            for j in range(self.Ny - 2):
                x = (i + 1) * self.dx
                y = 1 - (j + 1) * self.dy  # Inverting y to match problem description
                row = j * (self.Nx - 2) + i
                b[row] = self.get_source_term(x, y)
        
        return b
    
    def solve_steady_state(self):
        """
        Solve the steady-state heat equation
        """
        # Build coefficient matrix and RHS vector
        A = self.build_coefficient_matrix()
        b = self.build_rhs_vector()
        
        # Solve linear system
        T_interior = lin.solve(A, b)
        
        # Reconstruct full temperature field with boundary conditions
        T = np.zeros((self.Nx, self.Ny))
        T[1:-1, 1:-1] = T_interior.reshape((self.Nx-2, self.Ny-2))  # Remove inversion
        
        # Apply boundary conditions
        return self.apply_boundary_conditions(T)
    
    def solve_time_dependent(self, t_end, dt):
        """
        Solve the time-dependent heat equation (Case 4)
        
        Parameters:
        - t_end: Total simulation time
        - dt: Time step
        """
        if self.case != 4:
            raise ValueError("Time-dependent solver only for Case 4")
        
        # Initial condition with corrected y-axis
        x = np.linspace(0, 1, self.Nx)
        y = np.linspace(1, 0, self.Ny)  # Invert y-axis
        X, Y = np.meshgrid(x, y)
        T = np.sin(np.pi * X) * np.sin(np.pi * Y)
        
        # Time-stepping
        times = np.arange(0, t_end + dt, dt)
        solutions = [T.copy()]
        
        for _ in times[1:]:
            # Explicit finite difference method
            T_new = T.copy()
            T_new[1:-1, 1:-1] += self.alpha * dt * (
                (T[2:, 1:-1] - 2*T[1:-1, 1:-1] + T[:-2, 1:-1]) / self.dx**2 +
                (T[1:-1, 2:] - 2*T[1:-1, 1:-1] + T[1:-1, :-2]) / self.dy**2
            )
            
            # Enforce boundary conditions
            T_new[0, :] = 0
            T_new[-1, :] = 0
            T_new[:, 0] = 0
            T_new[:, -1] = 0
            
            T = T_new
            solutions.append(T.copy())
        
        return solutions, times
    
    def plot_solution(self, T, title=None):
        """
        Plot the temperature distribution
        """
        plt.figure(figsize=(8, 6))
        x = np.linspace(0, 1, self.Nx)
        y = np.linspace(1, 0, self.Ny)  # Corrected y-axis
        X, Y = np.meshgrid(x, y)
        
        plt.contourf(X, Y, T, cmap='viridis', levels=20)
        plt.colorbar(label='Temperature')
        plt.xlabel('x')
        plt.ylabel('y')
        if title:
            plt.title(title)
        plt.tight_layout()
        

        # Save the figure
        plt.savefig(f'{title.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
        

# Example usage remains the same as in previous implementation
# ```

# Key changes:
# 1. In `build_rhs_vector()`: `y = 1 - (j + 1) * self.dy` to invert y-axis
# 2. In `plot_solution()` and initial conditions: `y = np.linspace(1, 0, self.Ny)` to match problem description
# 3. Removed `.T` when reconstructing solution to maintain correct orientation

# These modifications ensure the axes align with the problem description, where y=0 is at the bottom and y=1 is at the top of the domain.

# Example usage
if __name__ == "__main__":
    # Case 1: Non-homogeneous source term
    solver1 = TwoDHeatEquationSolver(Nx=100, Ny=100, case=1)
    T1 = solver1.solve_steady_state()
    solver1.plot_solution(T1, 'Case 1: Non-homogeneous Source Term')
    
    # Case 2: Specific bottom boundary condition
    solver2 = TwoDHeatEquationSolver(Nx=50, Ny=50, case=2)
    T2 = solver2.solve_steady_state()
    solver2.plot_solution(T2, 'Case 2: Bottom Boundary Condition')
    
    # Case 3: Specific right boundary condition
    solver3 = TwoDHeatEquationSolver(Nx=50, Ny=50, case=3)
    T3 = solver3.solve_steady_state()
    solver3.plot_solution(T3, 'Case 3: Right Boundary Condition')
    
    # Case 4: Time-dependent solution
    solver4 = TwoDHeatEquationSolver(Nx=50, Ny=50, case=4, alpha=0.01)
    solutions, times = solver4.solve_time_dependent(t_end=0.1, dt=0.01)
    solver4.plot_solution(solutions[-1], 'Case 4: Time-Dependent Solution')
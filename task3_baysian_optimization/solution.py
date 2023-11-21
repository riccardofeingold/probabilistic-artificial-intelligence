"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
# import additional ...
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.

        self.number_data_points = 0
        self.data_points = []
        self.kappa = 4

        # mappings
        self.length_scales = [10, 1, 0.5]
        self.kernel_f_matern = Matern(nu=2.5, length_scale=self.length_scales[2])
        self.kernel_f_rbf = RBF(length_scale=self.length_scales[2])
        self.kernel_v_matern = DotProduct(sigma_0=0) + Matern(nu=2.5, length_scale=self.length_scales[2])
        self.kernel_v_rbf = DotProduct(sigma_0=0) + RBF(length_scale=self.length_scales[2])
        self.gp_f = GaussianProcessRegressor(kernel=self.kernel_f_matern, alpha=0.15, optimizer=None, random_state=0)
        self.gp_v = GaussianProcessRegressor(kernel=self.kernel_v_matern, alpha=0.0001, optimizer=None, random_state=0)

        # attributes for acquisition function
        self.beta = 1
        self.lambda_penalty = 2
        pass

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.

        if self.number_data_points == 0:
            next_point = np.random.uniform(0, 10)
        else:
            next_point = self.optimize_acquisition_function()
            
        return np.array(next_point).reshape(-1, 1)

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.
        mean_f, std_f = self.gp_f.predict(x, return_std=True)
        mean_v, std_v = self.gp_v.predict(x, return_std=True)
        
        # UCB
        x_f_next_ucb = mean_f + np.sqrt(self.beta) * std_f
        x_f_next_ucb -= self.lambda_penalty * np.maximum(mean_v, 0)
        
        return x_f_next_ucb
        # TODO: implement EI
        # EI
        # phi = norm(0, 1)
        # constraint_dist = norm(mean_v, std_v)
        # f_min = np.min([t["f"] for t in self.data_points])
        # z = (f_min - mean_f) / std_f
        # x_f_ei = std_f * (phi.cdf(z) * z + phi.pdf(z)) * constraint_dist.cdf(0) 

        # return x_f_ei.item()  

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.
        self.data_points.append({"x": x, "f": f, "v": v})
        
        self.number_data_points += 1

        x_values = np.array([t["x"] for t in self.data_points], dtype=np.float64)
        y_f = np.array([t["f"] for t in self.data_points], dtype=np.float64)
        y_v = np.array([t["v"] for t in self.data_points], dtype=np.float64)
        
        self.gp_f.fit(x_values.reshape(-1, 1), y_f.reshape(-1, 1))
        self.gp_v.fit(x_values.reshape(-1, 1), y_v.reshape(-1, 1))

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        x = np.array([t["x"] for t in self.data_points], dtype=np.float64)
        y_f = np.array([t["f"] for t in self.data_points], dtype=np.float64)
        y_v = np.array([t["v"] for t in self.data_points], dtype=np.float64)
        
        feasible_mask = y_v < self.kappa
        feasible_y_f = y_f[feasible_mask]
        feasible_x = x[feasible_mask]
        
        max_index = np.argmax(feasible_y_f)
        x_optimal = np.array(feasible_x[max_index])
                
        return x_optimal
        
    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        # TODO: Add plotting code
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.randn()
        cost_val = v(x) + np.random.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()

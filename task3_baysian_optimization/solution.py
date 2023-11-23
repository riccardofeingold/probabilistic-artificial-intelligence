"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
# import additional ...
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct, ConstantKernel
import matplotlib.pyplot as plt

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
        self.kernel_f_matern = Matern(nu=2.5, length_scale=0.5)
        self.kernel_f_rbf = ConstantKernel(0.5) * RBF(length_scale=0.5)

        self.linear_kernel = DotProduct(sigma_0=0)
        self.kernel_v_matern = self.linear_kernel + Matern(nu=2.5, length_scale=0.5)
        self.kernel_v_rbf = self.linear_kernel + ConstantKernel(np.sqrt(2)) * RBF(length_scale=0.5)

        self.gp_f = GaussianProcessRegressor(kernel=self.kernel_f_rbf, alpha=0.15, optimizer=None, random_state=0)
        self.gp_v = GaussianProcessRegressor(kernel=self.kernel_v_matern, alpha=0.0001, optimizer=None, random_state=0)

        # attributes for acquisition function
        self.beta = 1
        self.lambda_penalty = 30
        self.v_prior_mean = 4
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
            # next_point = np.random.uniform(0, 10)
            next_point = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 9, 10])
        else:
            next_point = self.optimize_acquisition_function()
            # d = self.get_distance(next_point)
            
            # if d <= 0.01:
            #     next_point = np.random.uniform(0, 10)
            
        return np.array(next_point).reshape(-1, 1)

    def get_distance(self, x):
        prev_points = np.array([t["x"] for t in self.data_points])
        distance = np.max([np.abs(x - p) for p in prev_points])
        return distance

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
        
        mean_v += self.v_prior_mean
        # UCB
        x_f_next_ucb = mean_f + np.sqrt(self.beta) * std_f
        x_f_next_ucb -= self.lambda_penalty * np.maximum(mean_v, 0)
        
        return x_f_next_ucb
        # TODO: implement EI
        # EI
        # epsilon = 0.5
        # phi = norm(0, 1)
        # # constraint_dist = norm(mean_v, std_v)
        # x = np.array([t["x"] for t in self.data_points], dtype=np.float64)
        # y_f = np.array([t["f"] for t in self.data_points], dtype=np.float64)
        # y_v = np.array([t["v"] for t in self.data_points], dtype=np.float64)
        
        # feasible_mask = y_v < self.kappa
        # feasible_y_f = y_f[feasible_mask]
        
        # max_index = np.argmax(feasible_y_f)
        # f_max = y_f[max_index]
        # z = (mean_f - f_max - epsilon) / std_f
        # # x_f_ei = std_f * (phi.cdf(z) * z + phi.pdf(z)) * constraint_dist.cdf(0) 
        # # x_f_ei = ((mean_f - f_max - epsilon)*phi.cdf(z) + std_f*phi.pdf(z)) * constraint_dist.cdf(0)
        # x_f_ei = ((mean_f - f_max - epsilon)*phi.cdf(z) + std_f*phi.pdf(z)) - self.lambda_penalty * np.maximum(mean_v, 0)
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
        y_v -= self.v_prior_mean
        
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

        # self.plot()
                
        return x_optimal
        
    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        # TODO: Add plotting code
        # Assume gp_f and gp_v are your trained Gaussian Process models for the objective and constraint
        # Let's also assume DOMAIN is your domain of interest, for example, np.array([[0, 10]])
        x = np.atleast_2d(np.linspace(DOMAIN[0, 0], DOMAIN[0, 1], 1000)).T

        # Predictions for objective and constraint functions
        y_pred_f, sigma_f = self.gp_f.predict(x, return_std=True)
        y_pred_v, sigma_v = self.gp_v.predict(x, return_std=True)

        # Find the optimal point
        opt_idx = np.argmax(y_pred_f)
        opt_point = x[opt_idx]
        opt_value = y_pred_f[opt_idx]

        # Plotting
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Objective function plot
        axs[0].fill_between(x.ravel(), y_pred_f.squeeze() - sigma_f, y_pred_f.squeeze() + sigma_f, alpha=0.1, color='k')
        axs[0].plot(x, y_pred_f.squeeze(), 'k', lw=1, label='Mean objective')
        axs[0].scatter(opt_point, opt_value, c='red', s=50, label='Optimal point')
        axs[0].set_title('Objective Function Posterior')
        axs[0].legend()

        # Constraint function plot
        axs[1].fill_between(x.ravel(), y_pred_v.squeeze() - sigma_v, y_pred_v.squeeze() + sigma_v, alpha=0.1, color='k')
        axs[1].plot(x, y_pred_v, 'k', lw=1, label='Mean constraint')
        axs[1].axhline(y=SAFETY_THRESHOLD, color='r', linestyle='--', label='Safety threshold')
        axs[1].fill_between(x.ravel(), y_pred_v.squeeze() - sigma_v, SAFETY_THRESHOLD, where=y_pred_v.squeeze() - sigma_v <= SAFETY_THRESHOLD, color='green', alpha=0.3, label='Safe region')
        axs[1].scatter(opt_point, self.gp_v.predict(opt_point.reshape(1, -1)), c='red', s=50, label='Optimal point')
        axs[1].set_title('Constraint Function Posterior')
        axs[1].legend()

        # Set common labels
        for ax in axs:
            ax.set_xlabel('Input domain')
            ax.set_ylabel('Output')
            ax.grid(True)

        # Show plot
        plt.tight_layout()
        # plt.show()
        plt.savefig("test.pdf")
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
        # assert x.shape == (1, DOMAIN.shape[0]), \
        #     f"The function next recommendation must return a numpy array of " \
        #     f"shape (1, {DOMAIN.shape[0]})"

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

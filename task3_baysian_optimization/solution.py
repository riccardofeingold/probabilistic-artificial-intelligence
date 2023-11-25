"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
# import additional ...
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct, ConstantKernel
import matplotlib.pyplot as plt
import os

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

        # mappings
        # TODO: 
        self.kernel_f_matern = Matern(nu=2.5, length_scale=0.5, length_scale_bounds=(0.5, 10))
        self.kernel_f_rbf = ConstantKernel(0.5) * RBF(length_scale=0.5, length_scale_bounds=(0.5, 10))

        self.kernel_v_matern = DotProduct(sigma_0=0) + Matern(nu=2.5, length_scale=0.5, length_scale_bounds=(0.5, 10))
        self.kernel_v_rbf = DotProduct(sigma_0=0) + ConstantKernel(np.sqrt(2)) * RBF(length_scale=0.5, length_scale_bounds=(0.5, 10))

        self.gp_f = GaussianProcessRegressor(kernel=self.kernel_f_rbf, alpha=0.15**2, optimizer=None, random_state=0)
        self.gp_v = GaussianProcessRegressor(kernel=self.kernel_v_matern, alpha=0.0001**2, optimizer=None, random_state=0)

        # attributes for acquisition function
        self.beta = 1
        self.lambda_penalty = 30
        self.v_prior_mean = 4
        self.af_type = "ucb"
        self.epsilon = 0.01
        # fixed the randomnes
        np.random.seed(42)
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

            # distances = self.distances(next_point, 10)
            # counter = 0
            # for d in distances:
            #     if d < 0.1:
            #         counter += 1
            
            # if counter > 4:
            #     noise = np.random.uniform(-1, 1)
            #     if noise >= 0:
            #         noise_clipped = np.clip(noise, 0.1, 1)
            #     else:
            #         noise_clipped = np.clip(noise, -0.1, -1)

            #     next_point += noise_clipped
            #     next_point = np.clip(noise_clipped, 0, 10)
            
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
        
        if self.af_type == "ucb":
            return self.UCB(x)
        elif self.af_type == "ei":
            return self.EI(x)
        elif self.af_type == "ts":
            return self.TS(x)
    
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
        self._get_solution()
        self.plot()

        return self.x_optimal
    
    def _get_solution(self):
        self.x = np.array([t["x"] for t in self.data_points], dtype=np.float64)
        self.y_f = np.array([t["f"] for t in self.data_points], dtype=np.float64)
        self.y_v = np.array([t["v"] for t in self.data_points], dtype=np.float64)
        
        # self.prob_v = np.array([self.cdf_of_constraint_function(value_x, value_v) for value_x, value_v in zip(self.x, self.y_v)])
        # feasible_mask = self.prob_v >= 0.95
        feasible_mask = self.y_v < SAFETY_THRESHOLD
        feasible_v = self.y_v[feasible_mask]
        
        feasible_y_f = self.y_f[feasible_mask]
        feasible_x = self.x[feasible_mask]
        
        max_index = np.argmax(feasible_y_f)
        self.x_optimal = feasible_x[max_index]
        self.v_optimal = feasible_v[max_index]
        self.y_optimal = feasible_y_f[max_index]

    def UCB(self, x):
        mean_f, std_f = self.gp_f.predict(x.reshape(-1, 1), return_std=True)
        mean_v, std_v = self.gp_v.predict(x.reshape(-1, 1), return_std=True)

        x_f_next_ucb = mean_f + np.sqrt(self.beta) * std_f
        #TODO: test new penalty version
        x_f_next_ucb -= self.lambda_penalty * np.maximum(mean_v, 0)

        return x_f_next_ucb
    
    def EI(self, x):
        mean_f, std_f = self.gp_f.predict(x.reshape(-1, 1), return_std=True)
        mean_v, std_v = self.gp_v.predict(x.reshape(-1, 1), return_std=True)

        self._get_solution()
            
        z = (self.y_optimal - mean_f - self.epsilon) / std_f
        prob_v = norm.cdf(SAFETY_THRESHOLD, mean_v, std_v)
        # x_f_ei = std_f * (norm.cdf(z, 0, 1) * z + norm.pdf(z, 0, 1)) * prob_v
        if prob_v >= 0.998:
            x_f_ei = std_f * (norm.cdf(z, 0, 1) * z + norm.pdf(z, 0, 1)) * prob_v
            x_f_ei = self.UCB(x)
        else:
            x_f_ei = prob_v
        
        return x_f_ei.item()
    
    def TS(self, x):
        sample_f = self.gp_f.sample_y(x)
        sample_v = self.gp_v.sample_y(x)

        return (sample_f - self.lambda_penalty * np.maximum(sample_v, 0)).item()

    def cdf_of_constraint_function(self, x, v):
        pred, std = self.gp_v.predict(x.reshape(-1, 1), return_std=True)
        return norm.cdf(SAFETY_THRESHOLD, pred, std)

    def distances(self, x, number=6):
        dps = np.array([dp["x"] for dp in self.data_points[-number:]])
        return np.array([x - dp for dp in dps])

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        # Assume gp_f and gp_v are your trained Gaussian Process models for the objective and constraint
        # Let's also assume DOMAIN is your domain of interest, for example, np.array([[0, 10]])
        x = np.atleast_2d(np.linspace(DOMAIN[0, 0], DOMAIN[0, 1], 1000)).T

        # Predictions for objective and constraint functions
        y_pred_f, sigma_f = self.gp_f.predict(x, return_std=True)
        y_pred_v, sigma_v = self.gp_v.predict(x, return_std=True)

        # Plotting
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Objective function plot
        axs[0].fill_between(x.ravel(), y_pred_f.squeeze() - sigma_f, y_pred_f.squeeze() + sigma_f, alpha=0.1, color='k')
        axs[0].plot(x, y_pred_f.squeeze(), 'k', lw=1, label='Mean objective')
        axs[0].scatter(self.x, self.y_f, c='black', s=50, label='Other data points')
        axs[0].scatter(self.x[0], self.y_f[0], c='red', s=50, label='Initial Point')
        axs[0].scatter(self.x_optimal, self.y_optimal, c='blue', s=50, label='Estimated Optimal Point')
        axs[0].set_title('Objective Function Posterior')
        axs[0].legend()

        # Constraint function plot
        axs[1].fill_between(x.ravel(), y_pred_v.squeeze() - sigma_v, y_pred_v.squeeze() + sigma_v, alpha=0.1, color='k')
        axs[1].plot(x, y_pred_v, 'k', lw=1, label='Mean constraint')
        axs[1].axhline(y=SAFETY_THRESHOLD, color='r', linestyle='--', label='Safety threshold')
        axs[1].fill_between(x.ravel(), y_pred_v.squeeze() - sigma_v, SAFETY_THRESHOLD, where=y_pred_v.squeeze() - sigma_v <= SAFETY_THRESHOLD, color='green', alpha=0.3, label='Safe region')
        axs[1].scatter(self.x, self.y_v, c='black', s=50, label='Other data points')
        axs[1].scatter(self.x[0], self.y_v[0], c='red', s=50, label='Initial point')
        axs[1].scatter(self.x_optimal, self.v_optimal, c='blue', s=50, label='Estimated Optimal Point')
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
        counter = 1
        extension = ".pdf"
        filename = f"plots_{counter}{extension}"
        full_path = os.path.join("/results/plots", filename)

        while os.path.exists(full_path):
            counter += 1
            filename = f"plots_{counter}{extension}"
            full_path = os.path.join("/results/plots", filename)

        plt.savefig(full_path)
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

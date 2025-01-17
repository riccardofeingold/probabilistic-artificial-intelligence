import os
import typing
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from matplotlib import cm

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)

        # TODO: Add custom initialization for your model here if necessary
        # kernel has been chosen based on the result of the 3-fold cross validation
        # hyperparameters have been found by maximising the marginal log liklihood
        self.kernel = Matern(length_scale=0.054, nu=1.5, length_scale_bounds=(1e-05, 100000.0)) + WhiteKernel(noise_level=0.00528, noise_level_bounds=(1e-07, 10.0))
        self.model = GaussianProcessRegressor(self.kernel, random_state=0, n_restarts_optimizer=1, normalize_y=True)

    def make_predictions(self, test_x_2D: np.ndarray, test_x_AREA: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of city_areas.
        :param test_x_2D: city_areas as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param test_x_AREA: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        # TODO: Use your GP to estimate the posterior mean and stddev for each city_area here
        gp_mean = np.zeros(test_x_2D.shape[0], dtype=float)
        gp_std = np.zeros(test_x_2D.shape[0], dtype=float)

        # TODO: Use the GP posterior to form your predictions here
        # obtain posterior mean and stdv
        gp_mean, gp_std = self.model.predict(test_x_2D, return_std=True)
        # adjust predictions based on test_x_area
        predictions = np.where(test_x_AREA == 1, gp_mean + gp_std, gp_mean)

        return predictions, gp_mean, gp_std

    def fitting_model(self, train_y: np.ndarray,train_x_2D: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_x_2D: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """

        # TODO: Fit your model here
        # using a simple random cluster: taking only 70% of the data for training => often used in ML
        self.train_x_2D, _, self.train_y, _ = train_test_split(train_x_2D, train_y, train_size=0.7, random_state=0)
        # fitting model
        self.model.fit(self.train_x_2D, self.train_y)

        pass

def model_selection(train_x: np.ndarray, train_y: np.ndarray, train_area: np.ndarray):
    # kernels to test
    kernels = [
        Matern(nu=0.5, length_scale=0.3),
        Matern(nu=0.5, length_scale=0.3) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-7, 1e-5)),
        Matern(nu=1.5, length_scale=0.1) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-7, 1e1)),
        Matern(nu=2.5, length_scale=0.1) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-7, 1e1)),
        RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-7, 1e1))
    ]

    # initialising a 3-fold cross validator
    kf = KFold(n_splits=3, shuffle=True, random_state=0)
    
    results = {}
    for k in kernels:
        results["Kernel: " + str(k)] = {}
        costs_per_fold = []
        optimal_thetas = []
        fold_counter = 1
        print("Kernel" + str(k))
        for train_indices, test_indices in kf.split(train_x):
            print("Fold " + str(fold_counter))
            # initialising gaussian process regressor model with one of the specified kernels
            model = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=1, normalize_y=True, random_state=0)

            # collect the fold for training and testing
            train_x_2D = train_x[train_indices]
            train_targets = train_y[train_indices]
            test_x_2D = train_x[test_indices]
            test_targets = train_y[test_indices]
            test_area = train_area[test_indices]

            # fitting model
            print("Fitting Model")
            model.fit(train_x_2D, train_targets)

            # get the posterior mean and stdv
            print("Prediction")
            gp_mean, gp_stdv = model.predict(test_x_2D, return_std=True)

            # to ensure no underprediction at areas == 1
            # I add the gp_stdv to the gp_mean
            predictions = np.where(test_area == 1, gp_mean + gp_stdv, gp_mean)

            # calculate the asymetric cost for this fold 
            print("Cost Calculation")
            cost = cost_function(test_targets, predictions, test_area)
            costs_per_fold.append(cost)
            optimal_thetas.append(model.kernel_.get_params())
            fold_counter += 1

        results["Kernel: " + str(k)] = {"mean cost": np.mean(costs_per_fold), "list cost": costs_per_fold, "optimal_thetas": optimal_thetas}

    print("Cross Validation DONE!")
    print(results)
    pass

# You don't have to change this function
def cost_function(ground_truth: np.ndarray, predictions: np.ndarray, AREA_idxs: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :param AREA_idxs: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask = (predictions < ground_truth) & [bool(AREA_idx) for AREA_idx in AREA_idxs]
    weights[mask] = COST_W_UNDERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)


# You don't have to change this function
def is_in_circle(coor, circle_coor):
    """
    Checks if a coordinate is inside a circle.
    :param coor: 2D coordinate
    :param circle_coor: 3D coordinate of the circle center and its radius
    :return: True if the coordinate is inside the circle, False otherwise
    """
    return (coor[0] - circle_coor[0])**2 + (coor[1] - circle_coor[1])**2 < circle_coor[2]**2

# You don't have to change this function 
def determine_city_area_idx(visualization_xs_2D):
    """
    Determines the city_area index for each coordinate in the visualization grid.
    :param visualization_xs_2D: 2D coordinates of the visualization grid
    :return: 1D array of city_area indexes
    """
    # Circles coordinates
    circles = np.array([[0.5488135, 0.71518937, 0.17167342],
                    [0.79915856, 0.46147936, 0.1567626 ],
                    [0.26455561, 0.77423369, 0.10298338],
                    [0.6976312,  0.06022547, 0.04015634],
                    [0.31542835, 0.36371077, 0.17985623],
                    [0.15896958, 0.11037514, 0.07244247],
                    [0.82099323, 0.09710128, 0.08136552],
                    [0.41426299, 0.0641475,  0.04442035],
                    [0.09394051, 0.5759465,  0.08729856],
                    [0.84640867, 0.69947928, 0.04568374],
                    [0.23789282, 0.934214,   0.04039037],
                    [0.82076712, 0.90884372, 0.07434012],
                    [0.09961493, 0.94530153, 0.04755969],
                    [0.88172021, 0.2724369,  0.04483477],
                    [0.9425836,  0.6339977,  0.04979664]])
    
    visualization_xs_AREA = np.zeros((visualization_xs_2D.shape[0],))

    for i,coor in enumerate(visualization_xs_2D):
        visualization_xs_AREA[i] = any([is_in_circle(coor, circ) for circ in circles])

    return visualization_xs_AREA

# You don't have to change this function
def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs_2D = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)
    visualization_xs_AREA = determine_city_area_idx(visualization_xs_2D)
    
    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.make_predictions(visualization_xs_2D, visualization_xs_AREA)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0

    # Plot the actual predictions
    fig, ax = plt.subplots()
    ax.set_title('Extended visualization of task 1')
    im = ax.imshow(predictions, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax = ax)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def extract_city_area_information(train_x: np.ndarray, test_x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the city_area information from the training and test features.
    :param train_x: Training features
    :param test_x: Test features
    :return: Tuple of (training features' 2D coordinates, training features' city_area information,
        test features' 2D coordinates, test features' city_area information)
    """
    train_x_2D = np.zeros((train_x.shape[0], 2), dtype=float)
    train_x_AREA = np.zeros((train_x.shape[0],), dtype=bool)
    test_x_2D = np.zeros((test_x.shape[0], 2), dtype=float)
    test_x_AREA = np.zeros((test_x.shape[0],), dtype=bool)

    #TODO: Extract the city_area information from the training and test features
    train_x_AREA = train_x[:, 2]
    train_x_2D = train_x[:, :2]
    test_x_AREA = test_x[:, 2]
    test_x_2D = test_x[:, :2]

    assert train_x_2D.shape[0] == train_x_AREA.shape[0] and test_x_2D.shape[0] == test_x_AREA.shape[0]
    assert train_x_2D.shape[1] == 2 and test_x_2D.shape[1] == 2
    assert train_x_AREA.ndim == 1 and test_x_AREA.ndim == 1

    return train_x_2D, train_x_AREA, test_x_2D, test_x_AREA

# you don't have to change this function
def main():
    # Load the training dateset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Extract the city_area information
    train_x_2D, train_x_AREA, test_x_2D, test_x_AREA = extract_city_area_information(train_x, test_x)

    # testing out different kernels using cross validation 
    # for each model the asymetric cost gets calcualted
    # the best model is selected based on the lowest cost
    print("Testing different models")
    model_selection(train_x_2D, train_y, train_x_AREA)
    ## Results from Cross Validation ##
    # results = {
    #     'Kernel: Matern(length_scale=0.3, nu=0.5)': 
    #     {
    #         'mean cost': 11.39948454309237, 
    #         'list cost': [11.745918099136956, 11.623824607841748, 10.828710922298404], 
    #         'optimal_thetas': [
    #             {'length_scale': 0.2943865454401839, 'length_scale_bounds': (1e-05, 100000.0), 'nu': 0.5}, 
    #             {'length_scale': 0.2918002701521217, 'length_scale_bounds': (1e-05, 100000.0), 'nu': 0.5}, 
    #             {'length_scale': 0.29577726391272985, 'length_scale_bounds': (1e-05, 100000.0), 'nu': 0.5}
    #         ]
    #     }, 
    #     'Kernel: Matern(length_scale=0.3, nu=0.5) + WhiteKernel(noise_level=1e-05)': 
    #     {
    #         'mean cost': 11.39944600833394, 
    #         'list cost': [11.745888081882084, 11.623774770851654, 10.828675172268083], 
    #         'optimal_thetas': [
    #             {'k1': Matern(length_scale=0.294, nu=0.5), 'k2': WhiteKernel(noise_level=1e-07), 'k1__length_scale': 0.29438841035118585, 'k1__length_scale_bounds': (1e-05, 100000.0), 'k1__nu': 0.5, 'k2__noise_level': 9.999999999999994e-08, 'k2__noise_level_bounds': (1e-07, 1e-05)}, 
    #             {'k1': Matern(length_scale=0.292, nu=0.5), 'k2': WhiteKernel(noise_level=1e-07), 'k1__length_scale': 0.29179968803203643, 'k1__length_scale_bounds': (1e-05, 100000.0), 'k1__nu': 0.5, 'k2__noise_level': 9.999999999999994e-08, 'k2__noise_level_bounds': (1e-07, 1e-05)}, 
    #             {'k1': Matern(length_scale=0.296, nu=0.5), 'k2': WhiteKernel(noise_level=1e-07), 'k1__length_scale': 0.29577906767706413, 'k1__length_scale_bounds': (1e-05, 100000.0), 'k1__nu': 0.5, 'k2__noise_level': 9.999999999999994e-08, 'k2__noise_level_bounds': (1e-07, 1e-05)}
    #         ]
    #     }, 
    #     'Kernel: Matern(length_scale=0.1, nu=1.5) + WhiteKernel(noise_level=1e-05)': 
    #     {
    #         'mean cost': 10.632530863767704, 
    #         'list cost': [11.34612123585776, 10.461970076031895, 10.089501279413454], 
    #         'optimal_thetas': [
    #             {'k1': Matern(length_scale=0.0537, nu=1.5), 'k2': WhiteKernel(noise_level=0.0052), 'k1__length_scale': 0.053668645145904595, 'k1__length_scale_bounds': (1e-05, 100000.0), 'k1__nu': 1.5, 'k2__noise_level': 0.005198630518695854, 'k2__noise_level_bounds': (1e-07, 10.0)}, 
    #             {'k1': Matern(length_scale=0.0533, nu=1.5), 'k2': WhiteKernel(noise_level=0.00533), 'k1__length_scale': 0.05333984972671367, 'k1__length_scale_bounds': (1e-05, 100000.0), 'k1__nu': 1.5, 'k2__noise_level': 0.005326642005857419, 'k2__noise_level_bounds': (1e-07, 10.0)}, 
    #             {'k1': Matern(length_scale=0.054, nu=1.5), 'k2': WhiteKernel(noise_level=0.00528), 'k1__length_scale': 0.054008649491161825, 'k1__length_scale_bounds': (1e-05, 100000.0), 'k1__nu': 1.5, 'k2__noise_level': 0.005280865082344792, 'k2__noise_level_bounds': (1e-07, 10.0)}
    #         ]
    #     }, 
    #     'Kernel: Matern(length_scale=0.1, nu=2.5) + WhiteKernel(noise_level=1e-05)': 
    #     {
    #         'mean cost': 10.83484473786235, 
    #         'list cost': [11.697141428241018, 10.559194693811303, 10.248198091534732], 
    #         'optimal_thetas': [
    #             {'k1': Matern(length_scale=0.0373, nu=2.5), 'k2': WhiteKernel(noise_level=0.00621), 'k1__length_scale': 0.03729185918765591, 'k1__length_scale_bounds': (1e-05, 100000.0), 'k1__nu': 2.5, 'k2__noise_level': 0.0062063429422440685, 'k2__noise_level_bounds': (1e-07, 10.0)}, 
    #             {'k1': Matern(length_scale=0.0372, nu=2.5), 'k2': WhiteKernel(noise_level=0.0064), 'k1__length_scale': 0.037234382780263545, 'k1__length_scale_bounds': (1e-05, 100000.0), 'k1__nu': 2.5, 'k2__noise_level': 0.0063971441379809466, 'k2__noise_level_bounds': (1e-07, 10.0)}, 
    #             {'k1': Matern(length_scale=0.0376, nu=2.5), 'k2': WhiteKernel(noise_level=0.00631), 'k1__length_scale': 0.03764235659619975, 'k1__length_scale_bounds': (1e-05, 100000.0), 'k1__nu': 2.5, 'k2__noise_level': 0.006314895151428851, 'k2__noise_level_bounds': (1e-07, 10.0)}
    #         ]
    #     }, 
    #     'Kernel: RBF(length_scale=1) + WhiteKernel(noise_level=1e-05)': 
    #     {
    #         'mean cost': 622.0409085220667, 
    #         'list cost': [621.0075988027727, 616.2934915800565, 628.821635183371], 
    #         'optimal_thetas': [
    #             {'k1': RBF(length_scale=1e-05), 'k2': WhiteKernel(noise_level=8.21e-05), 'k1__length_scale': 9.999999999999997e-06, 'k1__length_scale_bounds': (1e-05, 100000.0), 'k2__noise_level': 8.207869849551137e-05, 'k2__noise_level_bounds': (1e-07, 10.0)}, 
    #             {'k1': RBF(length_scale=1e-05), 'k2': WhiteKernel(noise_level=8.09e-05), 'k1__length_scale': 9.999999999999997e-06, 'k1__length_scale_bounds': (1e-05, 100000.0), 'k2__noise_level': 8.094872385980665e-05, 'k2__noise_level_bounds': (1e-07, 10.0)}, 
    #             {'k1': RBF(length_scale=1e-05), 'k2': WhiteKernel(noise_level=8.21e-05), 'k1__length_scale': 9.999999999999997e-06, 'k1__length_scale_bounds': (1e-05, 100000.0), 'k2__noise_level': 8.207873938341723e-05, 'k2__noise_level_bounds': (1e-07, 10.0)}
    #         ]
    #     }
    # }

    # Fit the model
    print('Fitting model')
    model = Model()
    model.fitting_model(train_y,train_x_2D)

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.make_predictions(test_x_2D, test_x_AREA)
    print(predictions)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()

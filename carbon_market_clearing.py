import numpy as np
from pathlib import Path
from tqdm import tqdm
from numba import njit, prange
import warnings

try:
    from .industry_model import IndustryModel
    from .input_processing import import_data
    from .power_model import PowerModel
    from .database import CarbonMarketDataBase
except ImportError:
    from industry_model import IndustryModel
    from input_processing import import_data
    from power_model import PowerModel
    from database import CarbonMarketDataBase

# Ignore warnings
warnings.filterwarnings("ignore")

input_path = Path("im_inputs")
output_path = Path("outputs")



class CapandTradeModel:
    def __init__(
            self, 
            db = CarbonMarketDataBase(), 
            learning_rate=0.01,
        ):
        
        # Initialize parameters
        self.learning_rate = learning_rate
        self.price_trajectory = db.initial_price_trajectory.squeeze().to_numpy()
        self.emissions_cap = db.supply.cap
        self.emissions = np.empty_like(db.supply.cap)
        
        # Initialize modules
        self.industry_model = IndustryModel(db=db.industry)
        self.power_model = PowerModel()

    def run_modules(self, p):
        # Implement the logic to run a module with the current price trajectory
        self.industry_model.solve(p=p)
        self.power_model.solve(p=p, industry_electricity_demand=None)
        return self.industry_model.emissions + self.power_model.emissions 

    def compute_gradient(self, ite):
        up_or_down = 1 + -2*(ite % 2)
        epsilon = self.price_trajectory.mean()*0.01*up_or_down # Small change in price
        gradient = np.empty_like(self.price_trajectory)
        
        # Calculate the change in total emissions for small changes in price
        total_emissions = (self.run_modules(self.price_trajectory) - self.emissions_cap).sum()
        for i in range(len(self.price_trajectory)):
            price_plus_epsilon = self.price_trajectory.copy()
            price_plus_epsilon[i] += max(price_plus_epsilon[i] + epsilon, 0)

            total_emissions_plus_epsilon = (self.run_modules(price_plus_epsilon) - self.emissions_cap).sum()
            change_in_emissions = total_emissions_plus_epsilon - total_emissions
            gradient[i] = (change_in_emissions / epsilon)
        
        gradient[np.isnan(gradient)] = 0 
        
        return gradient


    def update_price_trajectory(self, ite):
        gradient = self.compute_gradient(ite)
        for i in range(len(self.price_trajectory)):
            # Update the price trajectory based on the gradient
            self.price_trajectory[i] -= self.learning_rate * gradient[i]

    def run(self):
        max_iterations = 10
        some_tolerance = 0.01
        for ite in tqdm(range(max_iterations)):
            total_emissions = self.run_modules(self.price_trajectory) 
        
            if abs((total_emissions - self.emissions_cap).sum()) < some_tolerance:
                break
            else:
                self.update_price_trajectory(ite)
        self.emissions = total_emissions
    
if __name__ == "__main__":
    model = CapandTradeModel()
    final_price_trajectory = model.run()
    print(0)
import numpy as np
from itertools import product
try:
    from .input_processing import import_data
    from .utils import geometric_series
except ImportError:
    from input_processing import import_data
    from utils import geometric_series

START_YEAR = 2023
END_YEAR = 2050

data_dict, data_model, data_dict_aggregated = import_data()
allowance_price = np.full(END_YEAR - START_YEAR + 1, 100)

class IndustrySector():
    def __init__(
            self,
            region_sector_dict : dict = data_dict["denmark_cement"],
            start_year : int = 2023,
            end_year : int = 2050,
    ):  
        # Model parameters
        self.F = region_sector_dict["capex"]
        self.c = region_sector_dict["opex"]
        self.E = region_sector_dict["emission_intensities"]
        self.D = region_sector_dict["industrial_demand"]
        self.γ = region_sector_dict["γ"]
        self.L = int(region_sector_dict["asset_lifetime"])
        self.start_year = start_year
        self.end_year = end_year

        # Pre-process some parameter - β
        self.β = geometric_series(region_sector_dict["β"], self.L)

        # Model variables
        A_0 = region_sector_dict["asset_age"]
        A_shape = (A_0.shape[0], A_0.shape[1], end_year - start_year + 1)
        self.A = np.empty(A_shape)
        self.A[:,:,0] = A_0

        self.P = np.empty_like(self.c)
        self.C = np.empty_like(self.c)
        self.π = np.empty_like(self.c)
        self.annual_emissions = np.empty_like(self.E)
    
    def solve(self, p=allowance_price):
        # Extend allowance price array with its last value to calculate switching dynamics in the final modelling years
        p = np.concatenate([p, np.full(self.L, p[-1])])
        # Calculate the cost of each technology
        for t in range(self.end_year - self.start_year):
            # Compute current year emissions
            self.annual_emissions[:,t] = self.get_annual_emissions(t)
            self.C[:,t] = self.get_technology_cost(t, p[t:t+self.L])
            self.P[:,t] = self.get_switching_probability(t)
            a_L = self.get_annual_turnover(t)
            self.A[:,:,t+1] = self.update_A(t, a_L)
        self.annual_emissions[:,t+1] = self.get_annual_emissions(t)

    def get_technology_cost(self, t, p):
        # Extracting names in more readable format
        L=self.L
        β, c_t, E_t, F_t, = self.β, self.c[:,t:t+L], self.E[:,t:t+L], self.F[:,t]
        # Calculate the cost of each technology
        return F_t + β @ (c_t + E_t * p).T
    
    def get_annual_turnover(self,t):
        return self.A[:,self.L-1,t]
    
    def update_A(self, t, a_L):
        # Extract variable names
        A_t, P_t, L = self.A[:,:,t], self.P[:, t], self.L
        # Sum the total assets turning over in each year
        a_L_sum = a_L.sum()
        # Calculate the product of the switching probability and the total assets turning over
        product = a_L_sum * P_t
        # Reshape the product to be a column vector, to enable concatenation
        reshaped_product = product.reshape(8, 1)
        # Combine arrays
        return np.hstack((reshaped_product, A_t[:, :L-1]))

    def get_annual_emissions(self, t):
        # Extracting names in more readable format
        A_t, E_t, D_t = self.A[:,:,t], self.E[:,t], self.D[t]
        # Calculate the annual emissions
        return A_t.sum(axis=1)*E_t*D_t 
       
    def get_switching_probability(self, t):
        # Extracting names in more readable format
        C_t, γ = self.C[:,t], self.γ
        return np.exp(-C_t*(1/γ)) /  (np.exp(-C_t*(1/γ))).sum()
        
if __name__ == "__main__":
    model = IndustrySector()
    model.solve(allowance_price)
    model.solve(allowance_price*2)
import numpy as np
from itertools import product
from numba import float64, int64, njit, prange 
from numba.experimental import jitclass 
try:
    from .input_processing import import_data
    from .utils import geometric_series
    from .database import IndustryDataBase
except ImportError:
    from input_processing import import_data
    from utils import geometric_series
    from database import IndustryDataBase

START_YEAR = 2023
END_YEAR = 2050

data_dict, data_model, data_dict_aggregated = import_data()

class IndustrySector():
    def __init__(
            self,
            region_sector_dict : dict = data_dict["denmark_cement"],
            start_year : int = 2024,
            end_year : int = 2050,
    ):  
        # Model parameters
        self.F = region_sector_dict["capex"]
        self.c = region_sector_dict["opex"]
        self.E = region_sector_dict["emission_intensities"]
        self.D_initial = region_sector_dict["industrial_demand"]
        self.D = region_sector_dict["industrial_demand"].copy()
        self.γ = region_sector_dict["γ"]
        self.L = region_sector_dict["asset_lifetime"]
        self.ε = region_sector_dict["elasticities"]
        self.start_year = start_year
        self.end_year = end_year
        self.modelling_period = range(self.end_year - self.start_year + 1)

        # Pre-process some parameter - β
        self.β = geometric_series(region_sector_dict["β"], self.L)

        # Model variables
        A_0 = region_sector_dict["asset_age"]
        A_shape = (A_0.shape[0], A_0.shape[1], len(self.modelling_period))
        self.A = np.empty(A_shape)
        self.A[:,:,0] = A_0

        self.P = np.empty_like(self.c)
        self.C = np.empty_like(self.c)
        self.C_prime = np.empty_like(self.c)
        self.MAC = np.empty_like(self.c)
        self.π = np.empty_like(self.c)
        self.annual_emissions = np.empty_like(self.E[:,self.modelling_period])
    
    def solve(self, p):
        # Extend allowance price array with its last value to calculate switching dynamics in the final modelling years
        p = np.concatenate([p, np.full(self.L, p[-1])])
        # initiate demand
        self.D = self.D_initial.copy()
        # Calculate the cost of each technology
        for t in self.modelling_period:
            # Compute current year emissions
            self.annual_emissions[:,t] = self.get_annual_emissions(t)
            # Compute the cost of each technology, the prime cost is the cost without the carbon price
            # which we need to compute abtement costs for the MACC curves
            self.C[:,t], self.C_prime[:,t] = self.get_technology_cost(t, p[t:t+self.L])
            self.P[:,t] = self.get_switching_probability(t)
            # Update the asset age distribution
            if t < self.modelling_period[-1]:
                a_L = self.get_annual_turnover(t)
                self.A[:,:,t+1] = self.update_A(t, a_L)

        # Update the demand
        Δp_over_p = self.calculate_price_delta()

        # Apply the elasticities to the emissions
        ΔD = - (self.ε * Δp_over_p) * self.D[self.modelling_period]

        # Update the demand
        self.D[self.modelling_period] = self.D[self.modelling_period] + ΔD

        # Update the emissions
        for t in self.modelling_period:
            self.annual_emissions[:,t] = self.get_annual_emissions(t)

        # Calculate the MACC
        ΔE = self.calculate_emissions_intensity_change()
        self.MAC = self.calculate_marginal_abatement_cost(ΔE)
        self.abatement_potential = self.calculate_abatement_potential(ΔE)
    
    def calculate_emissions_intensity_change(self):
        E = self.E[:,self.modelling_period]
        # Calculate the y-o-y change in emissions intensity
        ΔE = np.max(E, axis=0) - E
        return ΔE

    def calculate_marginal_abatement_cost(self, ΔE):
        # Extracting names in more readable format
        modelling_period = self.modelling_period
        C_prime = self.C_prime[:,modelling_period]
        
        # Calculate the marginal abatement cost
        MAC = np.divide(C_prime - np.min(C_prime, axis=0), ΔE, out=np.zeros_like(C_prime), where=ΔE!=0)
        
        return MAC

    def calculate_abatement_potential(self,ΔE):
        # Extracting names in more readable format
        A, D = self.A, self.D[self.modelling_period]
        A_shifted = np.roll(self.A, -1, axis=2)
        # Calculate the y-o-y change in tech mix
        ΔA = A_shifted - A
        ΔA[:,:,-1] = 0
        Δtech_mix = ΔA.sum(axis=1)
        # Calculate the abatement potential
        return (Δtech_mix * ΔE)*D

    def calculate_price_delta(self):
        # Extracting names in more readable format
        modelling_period = self.modelling_period
        C = self.C[:,modelling_period]
        C_prime = self.C_prime[:,modelling_period]
        tech_mix = self.A.sum(axis=1)  
        β = self.β

        # Calculate the weighted average production cost per unit
        production_cost = ((C)*tech_mix).sum(axis=0)

        # Calculate the weighted average production cost per unit without the carbon price
        production_cost_no_carbon_price = ((
            self.C_prime[:,modelling_period] 
            )*tech_mix).sum(axis=0)

        # Calculate the price delta
        Δp_over_p = (
            production_cost - production_cost_no_carbon_price
            ) / production_cost

        return Δp_over_p
    
    def get_technology_cost(self, t, p):
        # Extracting names in more readable format
        L=self.L
        β, c_t, E_t, F_t, = self.β, self.c[:,t:t+L], self.E[:,t:t+L], self.F[:,t]
        # Calculate the levelized cost of each technology per tonne of output
        return (F_t + β @ (c_t + E_t * p).T) / β.sum(), (F_t + β @ (c_t).T) / β.sum()  
    
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
        reshaped_product = product.reshape(product.size, 1)
        # Combine arrays
        return np.hstack((reshaped_product, A_t[:, :L-1]))

    def get_annual_emissions(self, t):
        # Extracting names in more readable format
        A_t, E_t, D_t = self.A[:,:,t], self.E[:,t], self.D[t]
        # Calculate the annual emissions, in tonne
        # TODO: Adjust inputs such that we are consistent with the units of the 
        # emissions intensity and the demand and won't have to convert here
        return A_t.sum(axis=1)*E_t*D_t
       
    def get_switching_probability(self, t):
        # Extracting names in more readable format
        C_t, γ = self.C[:,t], self.γ
        return np.exp(-C_t*(1/γ)) /  (np.exp(-C_t*(1/γ))).sum()
        
if __name__ == "__main__":
    db = IndustryDataBase()
    model = IndustrySector(region_sector_dict=db.get_sector_region_data(sector="Cement", region="denmark"))
    allowance_price = np.ones(28)
    model.solve(allowance_price)
    model.solve(allowance_price*2)
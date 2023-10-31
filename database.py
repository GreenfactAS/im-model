import pandas as pd
import numpy as np
try:
    from .input_processing import import_data
    from .cost_model import CostModel
    from .utils import to_multidimensional_array
except ImportError:
    from input_processing import import_data
    from cost_model import CostModel
    from utils import to_multidimensional_array


class IndustryDataBase:
    def __init__(
        self
    ):
        # Import default data
        np_data_dict, data_model, data_dict = import_data()

        # Save data model
        self.data_model = data_model 

        # Save datasets - save datasets as numpy array to save memory and enable numba and jit
        self.emission_intensities = to_multidimensional_array(data_dict["emission_intensities"])
        self.industrial_demand = to_multidimensional_array(data_dict["industrial_demand"])
        self.γ = to_multidimensional_array(data_dict["γ"])
        self.β = to_multidimensional_array(data_dict["β"]) 
        self.asset_lifetime = to_multidimensional_array(data_dict["asset_lifetime"])
        self.initial_tech_mix = to_multidimensional_array(data_dict["initial_tech_mix"])
        self.asset_age = to_multidimensional_array(data_dict["asset_age"]) 

        # Create cost model object, to enable technology cost to vary between scenarios
        self.cost_model = CostModel(
            to_multidimensional_array(data_dict["other_opex"]),
            to_multidimensional_array(data_dict["commodity_use"]), 
            to_multidimensional_array(data_dict["commodity_prices"]),
            to_multidimensional_array(data_dict["capex"])
        )
        self.opex = self.cost_model.update_opex()
        self.capex = self.cost_model.update_capex()

    def get_sector_region_data(
            self,
            sector_idx : int,
            region_idx : int
    ) -> dict:
        """
        Get the data for a specific sector and region.

        Parameters:
        sector_idx (int): The index of the sector.
        region_idx (int): The index of the region.

        Returns:
        dict: A dictionary containing the data for the sector and region.
        """

        return {
            "emission_intensities": self.emission_intensities[sector_idx, region_idx, :],
            "industrial_demand": self.industrial_demand[sector_idx, region_idx, :],
            "γ": self.γ[sector_idx, region_idx, :],
            "β": self.β[sector_idx, region_idx, :],
            "asset_lifetime": self.asset_lifetime[sector_idx, region_idx, :],
            "initial_tech_mix": self.initial_tech_mix[sector_idx, region_idx, :],
            "asset_age": self.asset_age[sector_idx, region_idx, :],
            "opex": self.opex[sector_idx, region_idx, :],
            "capex": self.capex[sector_idx, region_idx, :]
                    }

    def update_opex(
            self,
            commodity_prices: pd.Series
    ) -> None:
        self.opex = self.cost_model.update_opex(commodity_prices)

if __name__ == "__main__":
    db = IndustryDataBase()
    print(0)

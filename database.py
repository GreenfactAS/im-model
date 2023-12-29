import pandas as pd
from pandas import IndexSlice as idx
import numpy as np
from itertools import product

try:
    from .input_processing import import_data
    from .cost_model import CostModel
    from .utils import to_multidimensional_array
except ImportError:
    from input_processing import import_data
    from cost_model import CostModel
    from utils import to_multidimensional_array

# Import default data
np_data_dict, data_model, data_dict = import_data()

class IndustryDataBase:
    def __init__(
        self, 
        data_dict: dict = data_dict,
        data_model: dict = data_model
    ):
        # Save data model
        self.data_model = data_model

        self.regions_indexed = {
            r : i for i, r in enumerate(data_model["regions"])
        }
        self.segments_indexed = {
            s : i for i, s in enumerate(data_model["segments"])
        }

        # Define the list of datasets to process
        datasets = [
            "emission_intensities",
            "industrial_demand",
            "γ",
            "β",
            "asset_lifetime",
            "asset_age",
            "elasticities"
        ]


        # define helper functions, to make the dictionary comprehension more readable
        check_type = lambda ds: ds.values \
            if isinstance(ds, pd.Series) or isinstance(ds, pd.DataFrame) \
            else ds
        get_dataset = lambda ds, r, s: data_dict[ds].loc[idx[r, s]]

        # Process each dataset using a dictionary comprehension
        for ds in datasets:
            setattr(self, ds, {
                r: {
                    s: check_type(get_dataset(ds, r, s)) 
                    for s in data_model["segments"]
                }
                for r  in data_model["regions"]
            })

        # Create cost model object, to enable technology cost to vary between scenarios
        self.cost_model = {
            s :  CostModel(
                to_multidimensional_array(data_dict["other_opex"].loc[idx[s]]),
                to_multidimensional_array(data_dict["commodity_use"].loc[idx[s]]), 
                to_multidimensional_array(data_dict["commodity_prices"]),
                to_multidimensional_array(data_dict["capex"].loc[idx[:,s,:]])
            ) for s in data_model["segments"]
        }
        
        self.opex = self.update_opex(to_multidimensional_array(data_dict["commodity_prices"]))
        self.capex = self.update_capex(to_multidimensional_array(data_dict["commodity_prices"]))

    def get_sector_region_data(
            self,
            sector : int,
            region : int
    ) -> dict:
        """
        Get the data for a specific sector and region.

        Parameters:
        sector (str): The sector.
        region (str): The region.

        Returns:
        dict: A dictionary containing the data for the sector and region.
        """

        return {
            "emission_intensities": self.emission_intensities[region][sector],
            "industrial_demand": self.industrial_demand[region][sector],
            "γ": self.γ[region][sector],
            "β": self.β[region][sector],
            "asset_lifetime": self.asset_lifetime[region][sector],
            "asset_age": self.asset_age[region][sector],
            "opex": self.opex[sector][self.regions_indexed[region]],
            "capex": self.capex[sector][self.regions_indexed[region]],
            "elasticities": self.elasticities[region][sector],
        }

    def update_opex(
            self,
            commodity_prices: pd.Series
    ) -> None:
        return {s : self.cost_model[s].update_opex(commodity_prices) for s in self.data_model["segments"]}
    
    def update_capex(
            self,
            commodity_prices: pd.Series
    ) -> None:
        return {s : self.cost_model[s].update_capex() for s in self.data_model["segments"]}

class SupplyDataBase:
    def __init__(
        self, 
        supply_data: pd.DataFrame = data_dict["supply"],
        data_model: dict = data_model
    ):
        self.cap = supply_data["cap"]
        self.msr_intake_rate = supply_data["msr_intake_rate"]
        self.msr_injection_rate = supply_data["msr_injection_rate"]
        self.msr_injection_amount = supply_data["msr_injection_amount"]
        self.msr_intake_rate = supply_data["msr_intake_rate"]
        self.tnac = supply_data["tnac"]  

class CarbonMarketDataBase:
    def __init__(
        self,
        industry_db = IndustryDataBase(),
        supply_db = SupplyDataBase(),
        initial_price_trajectory = data_dict["initial_price_trajectory"],
    ):
        self.industry = industry_db
        self.supply = supply_db
        self.initial_price_trajectory = initial_price_trajectory

if __name__ == "__main__":
    db = CarbonMarketDataBase()
    print(0)

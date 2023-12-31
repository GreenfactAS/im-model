import numpy as np
import pandas as pd
from functools import reduce
import os
from pathlib import Path
from itertools import product
try:
    from .utils import set_column_index_name
except ImportError:
    from utils import set_column_index_name
import yaml

def import_data() -> dict:
    """
    Import the data from the inputs folder and convert it to xarray format.
    The data is validated to ensure that it is consistent with the model.

    Returns
    -------
    dict
    """
    input_path = Path("im_inputs")

    # First import the ctrl file to get the list of parameters, 
    # and - importantly - the start year and end year for the simulation
    ctrl = pd.read_csv(
        input_path / "ctrl.csv", 
        index_col=["parameter"],
    ).squeeze()

    # Import the data model
    with open(input_path / "data_model.yaml") as file:
        data_model = yaml.load(file, Loader=yaml.FullLoader)
    
    start_year, end_year = data_model["start_year"], data_model["end_year"]
    modelling_period = [str(year) for year in range(start_year, end_year)]
    data_model["years"] = modelling_period

    data_dict = {
        "emission_intensities" : set_column_index_name(pd.read_csv(
            input_path / "emission_intensities.csv",
            index_col=["region", "segment", "technology"],
            usecols=["region", "segment", "technology"] + modelling_period,
            dtype={year : np.float64 for year in modelling_period},
            ).sort_index(), "year"),
        "industrial_demand" : set_column_index_name(pd.read_csv(
            input_path / "industrial_demand.csv",
            index_col=["region", "segment"],
            usecols=["region", "segment"] + modelling_period,
            dtype={year : np.float64 for year in modelling_period},
            ).sort_index(), "year"),
        "γ" : pd.read_csv(
            input_path / "gamma.csv",
            index_col=["region", "segment"],
            usecols=["gamma", "region", "segment"],
            dtype={"gamma" : np.float64},
            ).squeeze().sort_index(),
        "β" : pd.read_csv(
            input_path / "beta.csv",
            index_col=["region", "segment"],
            usecols=["beta", "region", "segment"],
            dtype={"beta" : np.float64},
            ).squeeze().sort_index(),
        "asset_lifetime" : pd.read_csv(
            input_path / "asset_lifetime.csv",
            index_col=["region", "segment"],
            usecols=["asset_lifetime", "region", "segment"],
            dtype={"asset_lifetime" : np.int8},
            ).squeeze().sort_index(),
        "capex" : set_column_index_name(pd.read_csv(
            input_path / "capex.csv",
            index_col=["region", "segment", "technology"],
            usecols=modelling_period + ["region", "segment", "technology"],
            dtype={year : np.float64 for year in modelling_period}
        ).sort_index(), "year"),
        "asset_age" : pd.read_csv(
            input_path / "asset_age.csv",
            index_col=["region", "segment", "technology"],
            usecols=[str(l) for l in range(1,26)] + ["region", "segment", "technology"],
            dtype={str(l) : np.float64 for l in range(1,26)}
        ).sort_index(),
        "elasticities" : pd.read_csv(
            input_path / "elasticities.csv",
            index_col=["region", "segment"],
            usecols=["elasticity", "segment", "region"],
            dtype={"elasticity" : np.float64}
        ).sort_index(), 
    }

    # Add the commodity prices and use data if it is available
    cost_data_dict = {
        "commodity_prices" : set_column_index_name(pd.read_csv(
            input_path / "commodity_prices.csv",
            index_col=["region", "commodity"],
            usecols=["region", "commodity"] + modelling_period,
            dtype={year : np.float64 for year in modelling_period}
        ).sort_index(), "year"),
        "commodity_use" : set_column_index_name(pd.read_csv(
            input_path / "commodity_use.csv",
            index_col=["segment", "technology", "commodity"],
            usecols=["segment", "technology", "commodity"] + modelling_period,
            dtype={year : np.float64 for year in modelling_period}
        ).sort_index(), "year"),
        "other_opex" : set_column_index_name(pd.read_csv(
            input_path / "other_opex.csv",
            index_col=["segment", "technology"],
            usecols=["segment", "technology"] + modelling_period,
            dtype={year : np.float64 for year in modelling_period}
        ).sort_index(), "year"),
    }

    # Add the supply data
    # need to check whether the supply data is consistent with the model: i.e., if the years are the same as for the other data
    supply_data = pd.read_csv(
        input_path / "supply.csv",
        index_col=["year"],
        usecols=[
            "year",
            "cap",
            "msr_intake_rate",
            "msr_injection_rate",
            "msr_injection_amount",
            "msr_intake_rate",
            "tnac",
        ],
    )

    initial_price_trajectory = pd.read_csv(
        input_path / "initial_price_trajectory.csv",
        index_col=["year"],
        usecols=["year", "price"],
    )

    # Validate the data
    InputValidation(data_dict, data_model).validate()

    # Add data for years after the modelling period beacuse the agents you be able to compute the cost of switching
    # technology in the final year of the simulation
    horizon = data_dict["asset_lifetime"].max()
    for name, data in data_dict.items():
        if isinstance(data, pd.DataFrame) and data.columns.name == "year":
            data_dict[name] = extrapolate_one_horizon_from_final_year(horizon, data)

    for name, data in cost_data_dict.items():
        if isinstance(data, pd.DataFrame) and data.columns.name == "year":
            cost_data_dict[name] = extrapolate_one_horizon_from_final_year(horizon, data)

    # Convert the data to numpy format, we do this to increase the speed of the model
    np_data_dict = {}
    for p in product(data_model["regions"], data_model["segments"]):
        # create region-segment index
        region_segment_str = "_".join(p).lower()
        # extract data for region-segment
        extract_data = lambda x : x.loc[p] if (isinstance(x.loc[p], np.float64) | isinstance(x.loc[p], np.int8)) else x.loc[p].values
        # extract data for region-segment
        np_data_dict[region_segment_str] = {
                    name : extract_data(data) for name, data in data_dict.items() if name != "elasticities"
                }
    
    # TO DO: return only one data dict, the second one is redundant
    pd_data_dict = {**data_dict, **cost_data_dict}
    pd_data_dict["supply"] = supply_data
    pd_data_dict["initial_price_trajectory"] = initial_price_trajectory
    np_data_dict = {**np_data_dict, **cost_data_dict}

    return np_data_dict, data_model, pd_data_dict

def extrapolate_one_horizon_from_final_year(horizon, data):
    last_data = data.iloc[:, -1]
    last_year = int(data.columns[-1])

    # Create a new DataFrame with additional years
    new_columns = [str(year) for year in range(last_year + 1, last_year + horizon + 1)]
    new_data = pd.DataFrame(data={col: last_data for col in new_columns}, index=data.index)

    # Concatenate the original DataFrame with the new DataFrame
    return pd.concat([data, new_data], axis=1)

class InputValidation:
    def __init__(self, data, data_model):
        """
        Parameters
        ----------
        data : dict
            The input data in xarray format
        data_model : dict
            The model specifiction
        """

        self.model = data_model
        self.data = data
    
    def validate(self):
        """
        Validate the input data

        Raises
        ------
        ValueError
            If the input data is not consistent with the model
        """
        self.validate_emission_intensities()
        self.validate_industrial_demand()
        self.validate_capex()
        self.validate_gamma()
        self.check_label_names()
    
    def validate_emission_intensities(self):
        # Check that the emission intensities are positive
        if (self.data["emission_intensities"] < 0).any().any():
            raise ValueError("Emission intensities must be positive")

    def validate_industrial_demand(self):
        # Check that the industrial demand is positive
        if (self.data["industrial_demand"] < 0).any().any():
            raise ValueError("Industrial demand must be positive")

    
    def validate_capex(self):
        # Check that the capex is positive
        if (self.data["capex"] < 0).any().any():
            raise ValueError("Capex must be positive")
    
    def validate_gamma(self):
        # Check that gamma is positive
        if (self.data["γ"] < 0).any():
            raise ValueError("Gamma must be positive")

    def check_label_names(self):
        # This function checks such that the labels in the data are consistent
        # with the labels in the model

        # Check that the regions, segements, year and technologies are the same, we do this in a clever way using a loop so as to not clutter the code
        labels_to_check = ["region", "segment", "year", "technology"]
        data_model_labels = {
            label : sorted([str(r).lower() for r in self.model[label + "s"]])
            for label in labels_to_check[0:3]
        }
        data_model_labels["technology"] = sorted([r.lower() for r in reduce(lambda x, y: x + y, self.model['technologies'].values())])
        for label in labels_to_check:
            for name, data in self.data.items():
                if label in data.index.names:            
                    if not np.array_equal(
                            data.index.get_level_values(label).astype(str).str.lower().drop_duplicates().sort_values().values, 
                            sorted(list(set(data_model_labels[label])))
                        ):
                        raise ValueError(
                            f"The {label} in the {name} data do not match the {label} in the data model"
                            )
                
if __name__ == '__main__':
    # Make sure we are in the right directory, the project root
    if "inputs" in os.getcwd(): 
        os.chdir("..")
    import_data()
    print(0)

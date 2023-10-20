import numpy as np
import pandas as pd
from itertools import product
try:
    from .input_processing import import_data
    from .utils import geometric_series
    from .sector_model import IndustrySector
except ImportError:
    from input_processing import import_data
    from utils import geometric_series
    from sector_model import IndustrySector

data_dict, data_model = import_data()
allowance_price = np.full(data_model["end_year"] - data_model["start_year"] + 1, 100)

class IndustryModel():
    def __init__(
            self,
            data_model : dict = data_model,
            data_dict : dict = data_dict,
            scenario_name="default_scenario"
    ):
        self.scenario_name = scenario_name  
        self.segments = data_model["segments"]
        self.regions = data_model["regions"]
        self.technologies = data_model["technologies"]
        self.raw_outputs = {}
        self.start_year = data_model["start_year"]
        self.end_year = data_model["end_year"]

    def solve(self, p=allowance_price, ):
        # saving the allowance price
        self.p = p
        # solving the model for each region and segment
        for region, segment in product(self.regions, self.segments):
            print(f"Solving {segment} sector in {region}...")
            sector_model = IndustrySector(
                region_sector_dict = data_dict[f"{region}_{segment}"],
                start_year = self.start_year,
                end_year = self.end_year,
            )
            sector_model.solve(p)
            self.raw_outputs[f"{region}_{segment}"] = sector_model
    
    def process_outputs(self):
        # Create a pd.DataFrame with the results
    
        # Creating indicies
        year_index = pd.Index(
                range(self.start_year, self.end_year+1),
                name = "year"
                )
    
        technology_mix_list = []
        industry_demand_list = []
        emissions_list = []

        for region, segment in product(self.regions, self.segments):
            sector_model = self.raw_outputs[f"{region}_{segment}"]

            # Creating indicies
            region_segment_technology_index = pd.MultiIndex.from_tuples(
                product(
                    [region],
                    [segment],
                    data_model["technologies"]["cement"]
                ),
                names = ["region", "segment", "technology"]
                )
            
            # Creating indicies
            region_segment_index = pd.MultiIndex.from_tuples(
                product(
                    [region],
                    [segment],
                    ["all"]
                ),
                names = ["region", "segment", "technology"]
                )
            
            # Technology mix
            technology_mix_regional_breakdown = pd.DataFrame(
                data=sector_model.A.sum(axis=1), 
                index=region_segment_technology_index, 
                columns=year_index
            )

            technology_mix_regional_breakdown_flat = technology_mix_regional_breakdown.stack()
            technology_mix_regional_breakdown_flat.name = "value"

            # Industry demand
            industry_demand = pd.DataFrame(
                data=sector_model.D[:len(year_index)], 
                index=year_index, 
                columns=region_segment_index 
            ).T

            industry_demand_flat = industry_demand.stack()
            industry_demand_flat.name = "value"

            # Emissions
            emissions = pd.DataFrame(
                data=sector_model.annual_emissions[:,:len(year_index)], 
                index=region_segment_technology_index, 
                columns=year_index
            )

            emissions_flat = emissions.stack()
            emissions_flat.name = "value"

            # Append to list
            technology_mix_list.append(technology_mix_regional_breakdown_flat)
            industry_demand_list.append(industry_demand_flat)
            emissions_list.append(emissions_flat)
        

        dummy_index = pd.MultiIndex.from_tuples(
                product(
                    ["all"],
                    ["all"],
                    ["all"],
                    range(self.start_year, self.end_year+1)
                ),
                names = ["region", "segment", "technology", "year"]
                )

        # Creating a flat version of the price
        price = pd.DataFrame(
            data=self.p[:len(year_index)], 
            index=dummy_index, 
            columns=["value"]
        )
    
        # Create a dictionary with the results
        processed_outputs_dict = {
            "technology_mix" : pd.concat(technology_mix_list),
            "industry_demand" : pd.concat(industry_demand_list),
            "emissions" : pd.concat(emissions_list),
            "allowance_price": price
        }

        processed_outputs = pd.concat(processed_outputs_dict, names="variable").reset_index()
        processed_outputs["scenario"] = self.scenario_name
    
        return processed_outputs
        
if __name__ == "__main__":
    model = IndustryModel()
    model.solve(allowance_price)
    processed_outputs = model.process_outputs()
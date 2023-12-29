import numpy as np
import pandas as pd
from itertools import product
try:
    from .input_processing import import_data
    from .utils import geometric_series, enumerated_product
    from .sector_model import IndustrySector
    from .cost_model import CostModel
    from .database import IndustryDataBase
except ImportError:
    from input_processing import import_data
    from utils import geometric_series, enumerated_product
    from sector_model import IndustrySector
    from cost_model import CostModel
    from database import IndustryDataBase

data_dict, data_model, data_dict_disaggregated = import_data()
allowance_price = pd.read_csv("im_inputs\initial_price_trajectory.csv", index_col=0).squeeze().to_numpy()

class IndustryModel():
    def __init__(
            self,
            data_model : dict = data_model,
            db = IndustryDataBase(),
            scenario_name="default_scenario"
    ):  
        self.db = db
        self.scenario_name = scenario_name  
        self.segments = data_model["segments"]
        self.regions = data_model["regions"]
        self.technologies = data_model["technologies"]
        self.raw_outputs = {}
        self.start_year = data_model["start_year"]
        self.end_year = data_model["end_year"]
        self.modelling_period = range(self.end_year - self.start_year + 1)

        # Aggregate model outputes
        self.emissions =- np.zeros(self.end_year - self.start_year + 1)

    def solve(
            self, 
            p=allowance_price, 
            commodity_prices=None,
            verbose=False
            ):
        
        # Adjust the allowance price to be of the right length, if necessary
        # if p is too long, just cut it off, if p is too short, extend it with the last value
        if len(p) > self.end_year - self.start_year + 1:
            p = p[:self.end_year - self.start_year + 1]
        elif len(p) < self.end_year - self.start_year + 1:
            p = np.concatenate([p, [p[-1]]*(self.end_year - self.start_year + 1 - len(p))])            

        # reset emissions trajctory
        self.emissions = np.zeros(self.end_year - self.start_year + 1)
        
        # saving the allowance price
        self.p = p

        # solving the model for each region and segment
        for region, segment in product(self.regions, self.segments):
            if verbose:
                print(f"Solving {segment} sector in {region}...")
            # Update the commodity prices
            sector_model = IndustrySector(
                region_sector_dict = self.db.get_sector_region_data(
                    sector = segment,
                    region = region
                ),
                start_year = self.start_year,
                end_year = self.end_year,
            )
            sector_model.solve(p)
            self.emissions = (self.emissions + sector_model.annual_emissions.sum(axis=0)[self.modelling_period])
            self.raw_outputs[f"{region}_{segment}"] = sector_model

    def generate_cost_curves(self):
        # Create a pd.DataFrame with the results
        price_levels = np.arange(0, 505, 5)  # Range of price levels from 5 to 500 in 5 euro increments

        # Compute emissions by sector and technology
        emissions_by_sector_technology = np.zeros((len(self.regions), len(self.segments), len(self.modelling_period), len(price_levels)))

        for i, price in enumerate(price_levels):
            print(f"Computing abatement potential for allowance price {price}...")
            self.solve(p=np.full(len(self.modelling_period), price))
            for j, sector in enumerate(self.segments):
                for k, region in enumerate(self.regions):
                    emissions_by_sector_technology[k, j,:, i] = self.raw_outputs[f"{region}_{sector}"].annual_emissions.sum(axis=0)
                
        # caluclate the abatement
        abatement_by_sector_technology = emissions_by_sector_technology[:,:,:,0][:,:,:,np.newaxis] - emissions_by_sector_technology
        abatement_by_sector_technology = np.diff(abatement_by_sector_technology, axis=2)

        # Create a multi-index DataFrame
        multi_index = pd.MultiIndex.from_product([
            self.regions, 
            self.segments, 
            range(self.start_year+1, self.end_year+1)], 
            names=['Region', 'Segment', "Year"])
        df_by_sector_technology_multi = pd.DataFrame(abatement_by_sector_technology.reshape(-1, len(price_levels)), index=multi_index, columns=price_levels)

        return df_by_sector_technology_multi
    
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
        mac_list = []
        abatement_potential_list = []

        for region, segment in product(self.regions, self.segments):
            sector_model = self.raw_outputs[f"{region}_{segment}"]

            # Creating indicies
            region_segment_technology_index = pd.MultiIndex.from_tuples(
                product(
                    [region],
                    [segment],
                    data_model["technologies"][segment]
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
            
            # Technology cost
            mac = pd.DataFrame(
                data=sector_model.MAC[:,:len(year_index)], 
                index=region_segment_technology_index, 
                columns=year_index
            )

            mac = mac.stack()
            mac.name = "value"

            abatement_potential = pd.DataFrame(
                data=sector_model.abatement_potential[:,:len(year_index)], 
                index=region_segment_technology_index, 
                columns=year_index
            )

            abatement_potential = abatement_potential.stack()
            abatement_potential.name = "value"

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
            mac_list.append(mac)
            abatement_potential_list.append(abatement_potential)

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
            "mac" : pd.concat(mac_list),
            "abatement_potential" : pd.concat(abatement_potential_list),
            "allowance_price": price["value"]
        }

        processed_outputs = pd.concat(
            processed_outputs_dict, 
            names=[
                "variable", 
                "region", 
                "segment", 
                "technology", 
                "year"
                ]
            ).reset_index()
        processed_outputs["scenario"] = self.scenario_name
    
        return processed_outputs
        
if __name__ == "__main__":
    model = IndustryModel()
    model.solve(allowance_price)
    model.process_outputs()
    model.generate_cost_curves()
    
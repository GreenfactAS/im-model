import numpy as np
import pandas as pd
try: 
    from .input_processing import import_data
    from .utils import to_multidimensional_array
except ImportError:
    from input_processing import import_data
    from utils import to_multidimensional_array

idx = pd.IndexSlice

class CostModel():
    def __init__(
        self,
        opex_other : np.ndarray,
        commodity_use : np.ndarray,
        commodity_prices : np.ndarray,
        capex : np.ndarray
    ): 
        self.opex_other = opex_other
        self.commodity_use = commodity_use
        self.commodity_prices = commodity_prices
        self.capex = capex
             
    def update_opex(self, commodity_prices=None, return_disaggregated_opex=False) -> np.array:
        """
        Update the opex attribute of the CostModel object.
        """
        if commodity_prices is None:
            P = self.commodity_prices
        else:
            P = commodity_prices

        Ο, U = self.opex_other, self.commodity_use
    
        if not return_disaggregated_opex:
            return np.einsum('ijk..., rjk...-> rik...', U, P) + Ο[np.newaxis, :, :]
        else:
            return np.einsum('ijk..., rjk...-> rijk...', U, P)
        
    def update_capex(self):
        return self.capex    
    
if __name__ == "__main__":
    # Testing
    np_data_dict, data_model, pd_data_dict = import_data()
    test_segment = data_model["segments"][0] 
    cost_model = CostModel(
        to_multidimensional_array(pd_data_dict["other_opex"].loc[idx[test_segment]]),
        to_multidimensional_array(pd_data_dict["commodity_use"].loc[idx[test_segment]]), 
        to_multidimensional_array(pd_data_dict["commodity_prices"]),
        to_multidimensional_array(pd_data_dict["capex"].loc[idx[:,test_segment,:]])
    )
    
    # Test update_capex
    disagg_opex = cost_model.update_opex(return_disaggregated_opex=True)
    lifetime = pd_data_dict["asset_lifetime"][:,test_segment][0]
    beta = pd_data_dict["β"][:,test_segment][0]
    other_opex = pd_data_dict["other_opex"].loc[idx[test_segment]]
    
    multiindex = pd.MultiIndex.from_product(
        [data_model["regions"], 
         data_model["technologies"][test_segment], 
         sorted(data_model["commodities"])]
    )
    # Create a list of years, starting with the years in the data model, and then the years after
    year_index = \
        [int(yy) for yy in data_model["years"]] \
    + [int(data_model["years"][-1]) + x for x in range(1, lifetime + 1)]

    df = pd.DataFrame(
        disagg_opex.reshape(-1, disagg_opex.shape[-1]), 
        index=multiindex, 
        columns=year_index
    )
    capex_copy = pd_data_dict["capex"].loc[idx[:,test_segment,:]].copy()*((1-beta)/(1-beta**lifetime))
    capex_copy["commodity"] = "Capex"
    capex_copy.set_index(
        "commodity", 
        append=True, 
        inplace=True
    )

    capex_copy.columns = year_index
    
    multiindex_dummy = pd.MultiIndex.from_product([ 
        data_model["technologies"][test_segment],
        data_model["regions"]
    ])

    multiindex_dummy.names = ["technology", "region"]
    dummy = pd.Series(index=multiindex_dummy, data=1)
    other_opex_with_regions = other_opex.multiply(dummy, axis="index")
    other_opex_with_regions = other_opex_with_regions.reorder_levels([1,0])
    other_opex_with_regions["commodity"] = "Other opex"
    other_opex_with_regions.set_index("commodity", append=True, inplace=True)
    other_opex_with_regions.columns = year_index
    all_costs = pd.concat([df, capex_copy, other_opex_with_regions])
    all_costs.index.names = ["region", "technology", "commodity"]

    df = all_costs.loc["austria",:][2024]
    df = df[df > 0].unstack("commodity")
    plt.figure(figsize=(10, 6))
    
    df.plot(kind="bar", stacked="True")
    # Plotting the stacked bar chart
    ax = df.plot(kind='bar', stacked=True)

    # Rotate the x-axis labels
    plt.xticks(rotation=0)

    # Optionally, adjust font size
    plt.xticks(fontsize=12)

    # Setting labels and title
    plt.xlabel('Technology', fontsize=12)
    plt.ylabel('TCO (EUR / tonne of product)', fontsize=12)
    plt.title('Technology cost', fontsize=12)
    plt.show()
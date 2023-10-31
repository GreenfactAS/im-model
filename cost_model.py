import numpy as np
import pandas as pd
try: 
    from .input_processing import import_data
    from .utils import to_multidimensional_array
except ImportError:
    from input_processing import import_data
    from utils import to_multidimensional_array

class CostModel():
    def __init__(
        self,
        opex_other : pd.DataFrame,
        commodity_use : pd.DataFrame,
        commodity_prices : pd.DataFrame,
        capex : pd.DataFrame
    ): 
        self.opex_other = opex_other
        self.commodity_use = commodity_use
        self.commodity_prices = commodity_prices
        self.capex = capex
             
    def update_opex(self, commodity_prices=None) -> np.array:
        """
        Update the opex attribute of the CostModel object.
        """
        if commodity_prices is None:
            P = self.commodity_prices
        else:
            P = commodity_prices

        Ο, U = self.opex_other, self.commodity_use
    
        return np.einsum('ijk..., rk...-> rij...', U, P) + Ο[np.newaxis, :, :, :]

    def update_capex(self):
        return self.capex
    
if __name__ == "__main__":
    # Testing
    data_dict, data_model = import_data()
    cost_model = CostModel(
        opex_other = data_dict["other_opex"],
        commodity_use = data_dict["commodity_use"],
        commodity_prices = data_dict["commodity_prices"]
                )
    print("0")
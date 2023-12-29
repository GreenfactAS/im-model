import pandas as pd
from pathlib import Path

input_path = Path("im_inputs")
output_path = Path("outputs")

class PowerModel():
    def __init__(self):
        self.emissions = pd.read_csv(
            input_path / "power_dummy_data.csv",
            usecols=["year", "power_emissions"],
            index_col="year"
            ).squeeze().to_numpy()
    
    def solve(self, industry_electricity_demand, p):
        # Solve the model
        pass
        
if __name__ == "__main__":
    model = PowerModel()
    industry_electricity_demand = 0
    p = 0
    model.solve(industry_electricity_demand, p)
    print(0)    
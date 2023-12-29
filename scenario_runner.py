from git import Repo
from pathlib import Path
import pandas as pd
import numpy as np
import datetime 
from itertools import product
try:
    from .industry_model import IndustryModel
    from .input_processing import import_data
    from .database import IndustryDataBase
except ImportError:
    from industry_model import IndustryModel
    from input_processing import import_data
    from database import IndustryDataBase

input_path = Path("im_inputs")
output_path = Path("outputs")

class ScenarioRunner:
    def __init__(self):
        self.models = []
        self.outputs = None

    def add_model_from_git(self):
        input_repo = Repo(input_path)
        # read in the file with commits to checkout
        commits = pd.read_csv("run_commits.csv")
        # checkout the commits
        for model in commits.index:
            input_repo.git.checkout(commits.loc[model, "commit"])
            # import the data
            # Import default data
            np_data_dict, data_model, data_dict = import_data()
            db = IndustryDataBase(data_dict=data_dict, data_model=data_model)
            # create the model
            model = IndustryModel(
                db=db,
                scenario_name=commits.loc[model, "scenario_name"]
                )
            # add the model to the list of models
            self.models.append(model)
    
    def add_shock(self, model):
        self.models.append(model)
    
    def add_default_model(self):
        data_dict, data_model = import_data()
        model = IndustryModel()
        self.models.append(model)
    
    def add_model(self, model):
        self.models.append(model)

    def run_all_models(self, allowance_prices_tests: dict = None):
        all_outputs = []

        if allowance_prices_tests is None:
            allowance_prices_tests = {
                "low": np.array([0]*25),
                "high": np.array([200]*25),
            }

        for model, price_trajcetory_name in product(self.models, allowance_prices_tests.keys()):
            model.solve(p=allowance_prices_tests[price_trajcetory_name])
            processed_results = model.process_outputs()
            processed_results["price_trajectory"] = price_trajcetory_name
            all_outputs.append(processed_results)
        all_outputs = pd.concat(all_outputs)

        # save file
        output_path = Path("outputs")
        output_path.mkdir(exist_ok=True)

        # get timstamp
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        # save file
        all_outputs.to_csv(output_path / f"{timestamp}_all_outputs.csv")

        self.outputs = all_outputs
        
        return all_outputs

if __name__ == "__main__":
    scenario_runner = ScenarioRunner()
    # create three databases with different gamma values
    scenario_runner.add_model_from_git()
    scenario_runner.run_all_models()
from git import Repo
from pathlib import Path
import pandas as pd

try:
    from .industry_model import IndustryModel
    from .input_processing import import_data
except ImportError:
    from industry_model import IndustryModel
    from input_processing import import_data

input_path = Path("im-inputs")    

class ScenarioRunner:
    def __init__(self):
        self.models = []

    def add_model_from_git(self, model):
        input_repo = Repo(input_path)
        # read in the file with commits to checkout
        commits = pd.read_csv(input_path / "commits.csv")
        # checkout the commits
        for model in commits.index:
            input_repo.git.checkout(commits.loc[model, "commit"])
            # import the data
            data_dict, data_model = import_data()
            # create the model
            model = IndustryModel(data_dict, data_model, sceanrio_name=commits.loc[model, "scenario_name"])
            # add the model to the list of models
            self.models.append(model)
    
    def add_shock(self, model):
        self.models.append(model)

    def run_all_models(self):
        for model in self.models:
            model.run()

if __name__ == "__main__":
    scenario_runner = ScenarioRunner()
    scenario_runner.add_model(IndustryModel())
    scenario_runner.run_all_models()

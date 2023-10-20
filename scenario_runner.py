from git import Repo

try:
    from .industry_model import IndustryModel
    from .input_processing import import_data
except ImportError:
    from industry_model import IndustryModel
    from input_processing import import_data

class ScenarioRunner:
    def __init__(self):
        self.models = []

    def add_model_from_git(self, model):
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

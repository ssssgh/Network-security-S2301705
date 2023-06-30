import argparse
from Experiment import Experiment
import json
from utils import AttributeAccessibleDict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Anomaly Detection Exp")
    parser.add_argument("--config_file", type=str, default=None)
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    config = AttributeAccessibleDict(config)

    # Load Exp class
    exp = Experiment(args=config)
    exp.start_experiment()
    

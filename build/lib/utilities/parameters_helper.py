"""
Helper functions for handling parameters data.
"""

import os
import json

from pathlib import Path
from typing import Dict, List, Union 

class Configs:
    """
    Class to handle the configuration of parameters data.
    """
    # Pre-define the configs dictionary format
    configs: Dict[str,Union[Path,Dict[str,Union[str,float,List[float]]]]]

    def __init__(self,
                 gsplat_config:str='scene003',
                 fout_config:str='track_cluttered',
                 mpc_config:str='flightroom', 
                 drone_config:str='carl',
                 control_config:str='body_rate_v1',
                 configs_path:Path=None,
                 gsplats_path:Path=None):
        """
        Initialize the Configs class to hold the various configs used in FiGS. By default these
        are the fout waypoints, MPC parameters, drone parameters, control parameters, and gsplats
        where the first four are loaded from JSON files in a configs directory and the gsplats are
        loaded from a YAML file within the gsplats directory. Default paths are provided for both.
        Users are able to add, remove, and list the configurations as needed and are welcome to
        create their own configurations as well to meet their needs of their trajectory generation
        pipeline.
        
        Args:
            - gsplat_config:  Name of the directory containing the gsplat configuration.
            - fout_config:    Name of the JSON file containing the fout waypoints.
            - mpc_config:     Name of the JSON file containing the MPC parameters.
            - drone_config:   Name of the JSON file containing the drone parameters.
            - control_config: Name of the JSON file containing the control parameters.
            - configs_path:   Path to the directory containing the JSON files.
            - gsplats_path:   Path to the directory containing the gsplats.

        Variables:
            - configs:        Dictionary containing the configs.
            - configs_path:   Path to the directory containing the JSON files.
            - gsplats_path:   Path to the directory containing the gsplats.
        """

        # Set the configuration directory
        if configs_path is None:
            self.configs_path = Path(__file__).parent.parent.parent/'configs'
        else:
            self.configs_path = configs_path

        # Set the gsplat directory
        if gsplats_path is None:
            self.gsplats_path = Path(__file__).parent.parent.parent/'gsplats'/'scene'/'output'
        else:
            self.gsplats_path = gsplats_path

        # Initialize the configs and gsplats dictionary
        self.configs = {}

        # Add the default configurations
        self.add_config('gaussian_splat',gsplat_config)
        self.add_config('fout_waypoints', fout_config)
        self.add_config('mpc_parameters', mpc_config)
        self.add_config('drone_parameters', drone_config)
        self.add_config('control_parameters', control_config)

    def add_config(self, key:str, value:str):
        """
        Add a configuration to the dictionary. The configuration is loaded from
        the config and/or gsplat directory.

        Args:
            - key:         Key to store the configuration under.
            - value:       Name of the JSON file to load the configuration from.
        """

        # Load the configuration from the relevant config directory
        if key == 'gaussian_splat':
            config = self.load_yaml(value)
        else:
            config = self.load_json(key,value)

        # Check if the key already exists in the dictionary
        if key in self.configs:
            print(f"Warning: The key '{key}' already exists in the configuration dictionary. Overwriting the existing configuration.")

        # Add the configuration to the dictionary
        self.configs[key] = config

    def remove_config(self, key):
        """
        Remove a configuration from the dictionary.

        Args:
            - key:         Key to remove from the dictionary.

        Returns:
            - config:      The configuration that was removed.
        """

        return self.configs.pop(key, None)
    
    def list_configs(self,Npad=70):
        """
        List the configurations currently loaded in the dictionary.
        """
        configs_list = list(self.configs.keys())
        Nstr = max([len(config) for config in configs_list])

        print("="*Npad)
        print("Currently Loaded Configs:")
        print("="*Npad)
        print("Configs:")
        for config in configs_list:
            param_category = config.ljust(Nstr)
            param_name = self.configs[config]["name"]
            print(f" - {param_category}: {param_name}")
        print("="*Npad)

    def get_config(self, key):
        """
        Get the parameters from the configuration dictionary.
        
        Args:
            - key:         Key to get the parameters for.

        Returns:
            - parameters:  Dictionary containing the parameters
        """

        if key not in self.configs:
            raise ValueError(f"The key '{key}' does not exist in the configuration dictionary.")
        
        return self.configs[key]
    
    def load_yaml(self, name:str):
        """
        Load a yaml from the gsplats directory.

        Args:
            - key:         Key to load the parameters for.

        Returns:
            - config:      Dictionary containing yaml configuration
        """
        search_path = self.gsplats_path/name
        yaml_configs = list(search_path.rglob("*.yml"))

        if len(yaml_configs) == 0:
            raise ValueError(f"The search path '{search_path}' did not return any configurations.")
        elif len(yaml_configs) > 1:
            raise ValueError(f"The search path '{search_path}' returned multiple configurations. Please specify a unique configuration within the directory.")
        else:
            config = {"name":name,"path":yaml_configs[0]}

        return config

    def load_json(self, key:str, value:str):
        """
        Load a json from the configs directory.
        
        Args:
            - key:         Key to load the parameters for.
            - value:       Name of the JSON file to load the configuration from.

        Returns:
            - config:      Dictionary containing the json configuration
        """
        json_config = self.configs_path/key/(value+".json")

        if not json_config.exists():
            raise ValueError(f"The json file '{json_config}' does not exist.")
        else:
            with open(json_config) as file:
                config = json.load(file)

        return config

    def __str__(self):
        """ 
        Return a string representation of the Configs object

        Returns:
            - str:         String representation of the Configs object
        """
        return str(self.configs)
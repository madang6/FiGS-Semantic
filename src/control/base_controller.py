from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Dict,List,Union
from pathlib import Path
import json

class BaseController(ABC):
    """
    Abstract base class for controllers.

    Methods:
        control(**inputs): Abstract method to be implemented by subclasses.
    
    Attributes:
        hz:      Frequency of the controller.
        nzcr:    Number of states in the controller.
    """
    def __init__(self,configs_path:Path=None) -> None:
        """
        Initialize the BaseController class.

        Args:
            configs_path: Path to the directory containing the JSON files.
        """
        # Set the configuration directory
        if configs_path is None:
            self.configs_path = Path(__file__).parent.parent.parent/'configs'
        else:
            self.configs_path = configs_path

        # Necessary attributes
        self.hz = None
        self.nzcr = None

    @abstractmethod
    def control(self, tcr:float,xcr:np.ndarray,
                upr:Union[None,np.ndarray],
                obj:Union[None,np.ndarray],
                icr:Union[None,np.ndarray],zcr:Union[None,torch.Tensor]):
        """
        Abstract control method to be implemented by subclasses.
        Args:
            **inputs: Key-value pairs representing the inputs.
        """
        pass

    def load_json_config(self, config:str, name:str):
        """

        """
        json_config = self.configs_path/config/(name+".json")

        if not json_config.exists():
            raise ValueError(f"The json file '{json_config}' does not exist.")
        else:
            # Load the json configuration
            with open(json_config) as file:
                config = json.load(file)
            
            # Add the name to the configuration
            config["name"] = name

        return config
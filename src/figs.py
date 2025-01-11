import json
import numpy as np

from render.gsplat import GSplat
from dynamics.flight import Flight
from control.base_controller import BaseController
from pathlib import Path
from typing import Dict,List,Type,Union
import time

class FiGS:
    """
    Class to handle the configuration of parameters data.
    """
    # Pre-define the configs dictionary format
    conFiGS: Dict[str,Union[Path,Dict[str,Union[str,float,List[float],Dict[str,Union[int,float]]]]]]

    def __init__(self,
                 scene_name:str='scene003',
                 rollout_type:str='baseline',
                 frame_name:str='carl',
                 configs_path:Path=None,gsplats_path:Path=None,verbose:bool=False) -> None:
        """
        FiGS facilitates drone flight within a GSplat and consists of two main components: a GSplat
        object and a Flight object. These configurations are stored in a dictionary called conFiGS.
        During initialization, the class loads a GSplat and sets up the conFiGS dictionary, while 
        the Flight object is kept as None until needed for simulation. To execute FiGS, you invoke
        the simulate method, which initializes the Flight object and runs the simulation. This design
        provides a streamlined interface with ACADOS by abstracting its JSON-based configuration and
        C backend.

        Args:
            - scene_name:     Name of the scene to load.
            - rollout_type:   Type of rollout to load.
            - frame_name:     Name of the frame to load.
            - configs_path:   Path to the directory containing the JSON files.
            - gsplats_path:   Path to the directory containing the gsplats.

        Variables:
            - conFiGS:        Dictionary containing the configurations used.
            - gsplat:         GSplat object.
            - flight:         Flight object.
            - configs_path:   Path to the directory containing the JSON files.
            - gsplats_path:   Path to the directory containing the gsplats.
        """

        # Set the configuration directory
        if configs_path is None:
            self.configs_path = Path(__file__).parent.parent/'configs'
        else:
            self.configs_path = configs_path

        # Set the gsplat directory
        if gsplats_path is None:
            self.gsplats_path = Path(__file__).parent.parent/'gsplats'/'workspace'/'outputs'
        else:
            self.gsplats_path = gsplats_path

        # Initialize the configurations dictionary and class objects
        self.conFiGS = {
            'gsplat': {
                'scene': None
            },
            'flight': {
                'rollout': None,
                'frame': None
            }
        }
        self.gsplat = None
        self.flight = None

        # Setup the GSplat and Flight objects
        self.load_gsplat(scene_name)
        self.load_flight(rollout_type,frame_name)

        # Print the configurations
        if verbose:
            self.list_configs()

    def load_gsplat(self, scene_name:str) -> None:
        """
        Fills the GSplat key in ConFiGS and generates the GSplat object.

        Args:
            - scene_name:     Name of the scene to load.
        """

        # Load the configurations
        scene_config = self.load_yaml_config(scene_name)

        # Add the configuration to the dictionary
        self.conFiGS['gsplat']['scene'] = scene_config

        # Create the GSplat object
        self.gsplat = GSplat(scene_config)

    def load_flight(self, rollout_type:str, frame_name:str) -> None:
        """
        Loads the Flight key in ConFiGS and generates the Flight object.

        Args:
            - rollout_type:   Type of rollout to load.
            - frame_name:     Name of the frame to load.
        """

        # Delete the flight object if it exists
        if self.flight is not None:
            self.flight.clear_generated_code()
            
        # Load the configurations
        rollout_config = self.load_json_config("rollout",rollout_type)
        frame_config = self.load_json_config("frame",frame_name)

        # Add the configuration to the dictionary
        self.conFiGS['flight']['rollout'] = rollout_config
        self.conFiGS['flight']['frame'] = frame_config

        # Create the Flight object
        self.flight = Flight(self.conFiGS['flight']['rollout'],self.conFiGS['flight']['frame'])

    def list_configs(self,Npad=70,npad=8) -> None:
        """
        Lists the currently loaded configurations.
        """
        components = list(self.conFiGS.keys())

        print("="*Npad)
        print("Currently Loaded Configs:")
        print("="*Npad)
        for component in components:
            print(component.capitalize())
            for key,value in self.conFiGS[component].items():
                print(f"  {key.ljust(npad)}: {value['name']}")
        print("="*Npad)

    def load_yaml_config(self, name:str) -> Dict[str,Union[str,Path]]:
        """
        Handles the yaml format for the conFiGS dictionary entries. Used for loading the GSplat

        Args:
            - name:   Name of the configuration to load.

        Returns:
            - config: Dictionary containing the configuration.
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

    def load_json_config(self, config:str, name:str) -> Dict[str,Union[str,float,List[float],Dict[str,Union[int,float]]]]:
        """
        Handles the json format for the conFiGS dictionary entries. Used for loading all configurations
        except the GSplat.

        Args:
            - config:   Name of the configuration to load.
            - name:     Name of the configuration to load.

        Returns:
            - config: Dictionary containing the configuration.
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
    
    def simulate(self,policy:Type[BaseController],
                 t0:float,tf:int,x0:np.ndarray,obj:Union[None,np.ndarray]=None,
                 cleanup:bool=True) -> None:
        """
        Simulates the flight using the given policy. The policy must be a subclass of BaseController.

        Args:
            - policy:   Policy to use for the simulation.
            - t0:       Initial time.
            - tf:       Final time.
            - x0:       Initial state.
            - obj:      Objective to use for the simulation.
            - cleanup:  Boolean to clear the flight object after simulation.
        """

        # Check if the Flight object exists, else reload it from the configurations
        if self.flight is None:
            self.flight = Flight(self.conFiGS['flight']['rollout'],self.conFiGS['flight']['frame'])

        # Simulate the flight
        Tro,Xro,Uro,Imgs,Tsol,Adv = self.flight.simulate(policy,self.gsplat,t0,tf,x0,obj)

        # Clear the Flight object if cleanup is True
        if cleanup:
            self.flight.clear_generated_code()
            self.flight = None

        return Tro,Xro,Uro,Imgs,Tsol,Adv
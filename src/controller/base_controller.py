from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Union

class BaseController(ABC):
    """
    Abstract base class for controllers.

    Methods:
        control(**inputs): Abstract method to be implemented by subclasses.
    
    Attributes:
        name: Name of the controller.
        hz:   Frequency of the controller.
    """
    def __init__(self, name:str, hz:int, nzcr:Union[int,None]=None) -> None:
        """
        Initialize the BaseController class.

        Args:
            name: Name of the controller.
            hz:   Frequency of the controller.
            nzcr: Size of latent space vector input (if applicable).
        """
        self.name = name
        self.hz = hz
        self.nzcr = nzcr

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

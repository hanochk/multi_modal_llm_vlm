import typing
from abc import ABC, abstractmethod
from PIL import Image
from typing import List
class ServiceInterface(ABC):

    def __init__(self):
        super().__init__() 

    @abstractmethod
    def get_url_response(self, url : str, text : List[str], *args):
        pass
    
    @abstractmethod
    def get_image_response(self, image : Image, text : List[str], *args):
        pass

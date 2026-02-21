from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Any

@dataclass
class Province:
    # Graph data
    index: int
    parent_region: Optional["Region"] # type: ignore
    plane: int
    name: Optional[str] = None
    coordinates: Optional[list] = None

    # Province properties
    terrain_int: int = 0
    has_gate: bool = False
    capital_location: bool = False
    capital_circle: bool = False
    capital_nation: Optional[int] = None

    # Province commands
    has_commands: bool = False
    poptype: Optional[int] = None
    owner: Optional[int] = None
    killfeatures: bool = False
    features: List[int] = field(default_factory=list)
    knownfeatures: List[int] = field(default_factory=list)
    fort: Optional[int] = None
    temple: bool = False
    lab: bool = False
    unrest: int = 0
    population: Optional[int] = None
    defence: Optional[int] = None
    skybox: Optional[str] = None
    batmap: Optional[str] = None
    groundcol: Optional[List[int]] = None
    rockcol: Optional[List[int]] = None
    fogcol: Optional[List[int]] = None

    # Drawing properties
    size: Optional[float] = None
    shape: Optional[float] = None
    height: Optional[int] = None

    def __str__(self):  # Printing the class returns this
        string = f'\nType - {type(self)}\n\n'
        for key, value in self.__dict__.items():
            string += f'{key} : {value}\n'
        return string

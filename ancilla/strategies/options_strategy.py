from typing import List, Optional
import bt

class OptionsStrategy(bt.Strategy):
    """
    A base class for options strategies. Inherit and extend this class to implement custom options logic.
    """

    def __init__(self, name: str, algos: List[bt.Algo], universe: Optional[List[str]] = None):
        print(f"Initializing OptionsStrategy: {name}")
        super().__init__(name, algos, universe)

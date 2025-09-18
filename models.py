from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod


@dataclass(frozen=True)
class MarketDataPoint:
    """Class for keeping track of an item in inventory."""
    timestamp: datetime
    symbol: str
    price: float

class Order:
    def __init__(self, symbol: str, quantity: int, price: float, status: str):
        self._symbol = symbol
        self._quantity = quantity
        self._price = price
        self._status = status

    # setter and getter 
    @property
    def symbol(self):
        return self._symbol
    @property
    def quantity(self):
        return self._quantity
    @property
    def price(self):
        return self._price
    @property
    def status(self):
        return self._status
    @status.setter
    def status(self, value):
        self._status = value


class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, tick: MarketDataPoint) -> list:
        pass

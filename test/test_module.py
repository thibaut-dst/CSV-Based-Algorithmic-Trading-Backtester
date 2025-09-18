import unittest
import dataclasses
from dataclasses import dataclass, FrozenInstanceError

class TestMutability(unittest.TestCase):
    """Test cases for demonstrating mutability differences between Order and MarketDataPoint"""
    
    def test_order_mutability(self):
        """Test that Order attributes can be modified after creation"""
        order = Order("AAPL", 100, 150.0, "PENDING")
        order.status = "FILLED"
        self.assertEqual(order.status, "FILLED")
    
    def test_market_data_immutability(self):
        """Test that MarketDataPoint is immutable"""
        data_point = MarketDataPoint(
            timestamp=datetime(2025, 9, 15, 16, 0, 0),
            symbol="AAPL",
            price=150.0
        )
        
        with self.assertRaises(dataclasses.FrozenInstanceError):
            data_point.price = 160.0

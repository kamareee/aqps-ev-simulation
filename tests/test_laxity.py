"""
Unit tests for laxity calculation and utility functions.
"""

import unittest
from src.aqps.data_structures import SessionInfo
from src.aqps.utils import (
    calculate_laxity,
    calculate_remaining_energy,
    energy_deliverable_at_rate,
    rate_needed_for_energy
)


class TestLaxityCalculation(unittest.TestCase):
    """Test laxity calculation function."""
    
    def setUp(self):
        """Set up common test data."""
        self.voltage = 220.0
        self.period_minutes = 5.0
    
    def test_positive_laxity(self):
        """Test session with plenty of time (positive laxity)."""
        session = SessionInfo(
            session_id="EV1",
            station_id="S1",
            arrival_time=0,
            departure_time=100,  # 100 periods
            energy_requested=20.0,  # 20 kWh
            max_rate=32.0,  # 32A
            current_charge=5.0  # Already has 5 kWh
        )
        
        # Remaining: 15 kWh
        # Max power: 32A * 220V = 7.04 kW
        # Energy per period: 7.04 * (5/60) = 0.587 kWh
        # Min periods needed: 15 / 0.587 ≈ 25.5
        # Available: 100 periods
        # Laxity: 100 - 25.5 ≈ 74.5
        
        laxity = calculate_laxity(session, 0, self.voltage, self.period_minutes)
        self.assertGreater(laxity, 70)
        self.assertLess(laxity, 80)
    
    def test_negative_laxity(self):
        """Test session with insufficient time (negative laxity)."""
        session = SessionInfo(
            session_id="EV2",
            station_id="S2",
            arrival_time=0,
            departure_time=10,  # Only 10 periods
            energy_requested=50.0,  # 50 kWh (lots of energy)
            max_rate=32.0,
            current_charge=0.0
        )
        
        laxity = calculate_laxity(session, 0, self.voltage, self.period_minutes)
        self.assertLess(laxity, 0)  # Insufficient time
    
    def test_zero_remaining_demand(self):
        """Test fully charged EV (should have infinite laxity)."""
        session = SessionInfo(
            session_id="EV3",
            station_id="S3",
            arrival_time=0,
            departure_time=50,
            energy_requested=20.0,
            max_rate=32.0,
            current_charge=20.0  # Fully charged
        )
        
        laxity = calculate_laxity(session, 0, self.voltage, self.period_minutes)
        self.assertEqual(laxity, float('inf'))
    
    def test_zero_max_rate(self):
        """Test EV with zero max rate (should have -inf laxity)."""
        session = SessionInfo(
            session_id="EV4",
            station_id="S4",
            arrival_time=0,
            departure_time=50,
            energy_requested=20.0,
            max_rate=0.0,  # Cannot charge
            current_charge=0.0
        )
        
        laxity = calculate_laxity(session, 0, self.voltage, self.period_minutes)
        self.assertEqual(laxity, float('-inf'))
    
    def test_laxity_decreases_with_time(self):
        """Test that laxity decreases as time passes."""
        session = SessionInfo(
            session_id="EV5",
            station_id="S5",
            arrival_time=0,
            departure_time=100,
            energy_requested=20.0,
            max_rate=32.0,
            current_charge=0.0
        )
        
        laxity_t0 = calculate_laxity(session, 0, self.voltage, self.period_minutes)
        laxity_t10 = calculate_laxity(session, 10, self.voltage, self.period_minutes)
        laxity_t20 = calculate_laxity(session, 20, self.voltage, self.period_minutes)
        
        self.assertGreater(laxity_t0, laxity_t10)
        self.assertGreater(laxity_t10, laxity_t20)
        self.assertAlmostEqual(laxity_t0 - laxity_t10, 10.0, places=1)


class TestEnergyCalculations(unittest.TestCase):
    """Test energy-related utility functions."""
    
    def test_remaining_energy(self):
        """Test remaining energy calculation."""
        session = SessionInfo(
            session_id="EV1",
            station_id="S1",
            arrival_time=0,
            departure_time=100,
            energy_requested=50.0,
            max_rate=32.0,
            current_charge=30.0
        )
        
        remaining = calculate_remaining_energy(session)
        self.assertAlmostEqual(remaining, 20.0)
    
    def test_remaining_energy_fully_charged(self):
        """Test remaining energy when fully charged."""
        session = SessionInfo(
            session_id="EV2",
            station_id="S2",
            arrival_time=0,
            departure_time=100,
            energy_requested=50.0,
            max_rate=32.0,
            current_charge=50.0
        )
        
        remaining = calculate_remaining_energy(session)
        self.assertEqual(remaining, 0.0)
    
    def test_energy_deliverable(self):
        """Test energy deliverable calculation."""
        # 32A at 220V for 1 hour (12 periods of 5 minutes)
        # Power = 32 * 220 / 1000 = 7.04 kW
        # Energy = 7.04 kWh
        energy = energy_deliverable_at_rate(32.0, 12, 220.0, 5.0)
        self.assertAlmostEqual(energy, 7.04, places=2)
    
    def test_rate_needed(self):
        """Test rate needed calculation."""
        # Need 10 kWh in 2 hours (24 periods)
        # Power = 10 / 2 = 5 kW
        # Rate = 5000 / 220 ≈ 22.73A
        rate = rate_needed_for_energy(10.0, 24, 220.0, 5.0)
        self.assertAlmostEqual(rate, 22.73, places=1)


if __name__ == '__main__':
    unittest.main()

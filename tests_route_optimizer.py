"""Standalone unit checks for LifeLink-AI route optimisation."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('lifelink_transport', Path(__file__).parent/'app'/'transport.py')
transport=importlib.util.module_from_spec(spec); spec.loader.exec_module(transport)

class RouteOptimizerTests(unittest.TestCase):
    def test_fastest_route_can_differ_from_shortest_distance(self):
        result=transport.find_fastest_route('Pune','Mumbai')
        self.assertEqual(result['distance_km'],380.0)
        self.assertEqual(result['nodes'],['Pune','Nashik','Mumbai'])
        direct=next(x for x in result['alternatives'] if x['nodes']==['Pune','Mumbai'])
        self.assertLess(direct['distance_km'],result['distance_km'])
        self.assertGreater(direct['travel_hours'],result['travel_hours'])

    def test_feasibility_remaining_time(self):
        result=transport.check_transport_feasibility('Kidney',2.0,0)
        self.assertEqual(result['maximum_minutes'],180)
        self.assertEqual(result['estimated_minutes'],120)
        self.assertEqual(result['remaining_minutes'],60)
        self.assertTrue(result['feasible'])

    def test_harvest_time_reduces_available_window(self):
        result=transport.check_transport_feasibility('Kidney',1.1,2.0)
        self.assertEqual(result['remaining_minutes'],-6.0)
        self.assertFalse(result['feasible'])

class TransportScenarioTests(unittest.TestCase):
    def test_mode_comparison_returns_fastest_air_option(self):
        rows=transport.compare_transport_modes(2.0)
        self.assertEqual(rows[0]['mode'],'AIR_AMBULANCE')
        self.assertLess(rows[0]['travel_hours'],2.0)

    def test_high_traffic_scenario_recalculates(self):
        normal=transport.route_scenario('Pune','Mumbai','LOW')
        heavy=transport.route_scenario('Pune','Mumbai','VERY_HIGH')
        self.assertGreaterEqual(heavy['travel_hours'],normal['travel_hours'])

if __name__=='__main__': unittest.main(verbosity=2)

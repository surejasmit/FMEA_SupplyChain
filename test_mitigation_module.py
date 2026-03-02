"""
Test script for Supply Chain Risk Mitigation Module
Demonstrates all capabilities without UI
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from mitigation_module import TransportOptimizer, DisruptionExtractor
from mitigation_module.network_config import validate_network, ROUTE_MAP


def test_import_independence():
    """Ensure OCR availability is unaffected by dynamic_network import failures."""
    import importlib, sys, builtins

    # Capture original state
    import mitigation_module.disruption_extractor as de
    orig_ocr = de.OCR_AVAILABLE

    # Prepare a fake import that fails for dynamic_network
    real_import = builtins.__import__
    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.endswith('dynamic_network'):
            raise ImportError("simulated failure")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = fake_import
    try:
        # force reload to re-run import logic under patched importer
        de_reloaded = importlib.reload(de)
        # dynamic routing should be disabled
        assert not de_reloaded.DYNAMIC_ROUTING_AVAILABLE
        # OCR availability should remain based solely on easyocr import
        assert de_reloaded.OCR_AVAILABLE == orig_ocr
    finally:
        builtins.__import__ = real_import

    print("✅ import independence verified")


def test_network_validation():
    """Test 1: Network Configuration"""
    print("=" * 60)
    print("TEST 1: Network Validation")
    print("=" * 60)
    
    info = validate_network()
    print(f"✓ Total Routes: {info['num_routes']}")
    print(f"✓ Warehouses: {info['num_warehouses']}")
    print(f"✓ Clients: {info['num_clients']}")
    print(f"✓ Total Supply: {info['total_supply']} units")
    print(f"✓ Total Demand: {info['total_demand']} units")
    print(f"✓ Surplus: {info['surplus']} units")
    print()


def test_text_extraction():
    """Test 2: Text-Based Disruption Extraction"""
    print("=" * 60)
    print("TEST 2: Text Extraction")
    print("=" * 60)
    
    extractor = DisruptionExtractor()
    
    test_text = "Major flooding in Boston area. I-95 highway closed. Port operations suspended."
    
    events = extractor.extract_from_text(test_text)
    
    print(f"Input: {test_text}")
    print(f"✓ Extracted {len(events)} disruption event(s):")
    for i, event in enumerate(events, 1):
        print(f"\n  Event {i}:")
        print(f"    Route ID: {event.target_route_id}")
        print(f"    Impact Type: {event.impact_type}")
        print(f"    Cost Multiplier: {event.cost_multiplier}×")
        print(f"    Severity: {event.severity_score}/10")
    print()


def test_optimization_baseline():
    """Test 3: Baseline Optimization (No Disruptions)"""
    print("=" * 60)
    print("TEST 3: Baseline Optimization")
    print("=" * 60)
    
    optimizer = TransportOptimizer()
    result = optimizer.solve(use_disrupted_costs=False)
    
    if result['status'] == 'success':
        print(f"✓ Optimization successful!")
        print(f"✓ Total Cost: ${result['total_cost']:,.2f}")
        print(f"\n✓ Optimal Flow (x_ij) - Top 5 Routes:")
        
        # Sort by flow value
        flows = sorted(result['flows'].items(), key=lambda x: x[1], reverse=True)
        for route_id, flow in flows[:5]:
            src, dest = ROUTE_MAP[route_id]
            print(f"    Route {route_id} ({src} → {dest}): {flow:.0f} units")
    else:
        print(f"✗ Optimization failed: {result['message']}")
    print()


def test_disruption_optimization():
    """Test 4: Optimization with Disruptions"""
    print("=" * 60)
    print("TEST 4: Disruption-Adjusted Optimization")
    print("=" * 60)
    
    # Simulate disruptions
    disruptions = [
        {
            'target_route_id': 1,  # Boston route
            'impact_type': 'flood',
            'cost_multiplier': 2.5,
            'severity_score': 8
        },
        {
            'target_route_id': 2,  # New York route
            'impact_type': 'strike',
            'cost_multiplier': 1.8,
            'severity_score': 6
        }
    ]
    
    print("Simulated Disruptions:")
    for d in disruptions:
        print(f"  - Route {d['target_route_id']}: {d['impact_type']} (×{d['cost_multiplier']})")
    print()
    
    optimizer = TransportOptimizer()
    optimizer.apply_disruptions(disruptions)
    
    result = optimizer.solve(use_disrupted_costs=True)
    
    if result['status'] == 'success':
        print(f"✓ Optimization successful!")
        print(f"✓ Adjusted Total Cost: ${result['total_cost']:,.2f}")
    else:
        print(f"✗ Optimization failed: {result['message']}")
    print()


def test_plan_comparison():
    """Test 5: Compare Original vs. Risk-Adjusted Plans"""
    print("=" * 60)
    print("TEST 5: Plan Comparison (Original vs. Adjusted)")
    print("=" * 60)
    
    # Simulate major disruption
    disruptions = [
        {'target_route_id': 1, 'impact_type': 'flood', 'cost_multiplier': 3.0, 'severity_score': 9},
        {'target_route_id': 6, 'impact_type': 'hurricane', 'cost_multiplier': 2.5, 'severity_score': 8}
    ]
    
    optimizer = TransportOptimizer()
    optimizer.apply_disruptions(disruptions)
    
    comparison = optimizer.compare_plans()
    
    if comparison['status'] == 'success':
        print("✓ Comparison successful!")
        print(f"\n📊 Cost Analysis:")
        print(f"   Original Plan:      ${comparison['original_cost']:,.2f}")
        print(f"   Risk-Adjusted Plan: ${comparison['adjusted_cost']:,.2f}")
        print(f"   Cost Increase:      ${comparison['cost_delta']:,.2f} ({comparison['cost_delta_pct']:.1f}%)")
        
        if comparison['flow_changes']:
            print(f"\n📦 Significant Flow Changes ({len(comparison['flow_changes'])} routes):")
            for route_id, changes in list(comparison['flow_changes'].items())[:5]:
                src, dest = ROUTE_MAP[route_id]
                print(f"   Route {route_id} ({src} → {dest}):")
                print(f"      {changes['original']:.0f} → {changes['adjusted']:.0f} units ({changes['delta']:+.0f}, {changes['delta_pct']:+.1f}%)")
    else:
        print(f"✗ Comparison failed")
    print()


def test_route_details():
    """Test 6: Route Details with Costs"""
    print("=" * 60)
    print("TEST 6: Route Cost Details")
    print("=" * 60)
    
    disruptions = [
        {'target_route_id': 3, 'impact_type': 'accident', 'cost_multiplier': 1.5, 'severity_score': 5}
    ]
    
    optimizer = TransportOptimizer()
    optimizer.apply_disruptions(disruptions)
    
    df = optimizer.get_route_details()
    
    print("✓ Route Cost Summary:")
    print(df.to_string(index=False))
    print()


def run_all_tests():
    """Run complete test suite"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Supply Chain Risk Mitigation Module - Test Suite".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    try:
        test_network_validation()
        test_text_extraction()
        test_optimization_baseline()
        test_disruption_optimization()
        test_plan_comparison()
        test_route_details()
        
        print("=" * 60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print()
        print("Next Steps:")
        print("1. Run Streamlit dashboard: streamlit run app.py")
        print("2. Navigate to 🚚 Supply Chain Risk tab")
        print("3. Try different input methods (Text, CSV, Image, News)")
        print()
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()

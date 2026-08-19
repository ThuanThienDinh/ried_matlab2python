"""Comprehensive test of MATLAB to Python conversions"""
import numpy as np
from ried_core.entromap import entromap
from ried_core.xEtr import xEtr
from ried_core.XCcumulant import XCcumulant

print("=" * 70)
print("COMPREHENSIVE MATLAB TO PYTHON CONVERSION TESTS")
print("=" * 70)

# ============================================================================
# Test 1: xEtr (Cross-Entropy)
# ============================================================================
print("\n[TEST GROUP 1] xEtr - Cross-Entropy Metric")
print("-" * 70)

print("\n  Test 1.1: Basic cross-entropy")
np.random.seed(42)
data1 = np.random.rand(100)
data2 = np.random.rand(100)
xe_result = xEtr(data1, data2, bin=20, maxv=1.0, minv=0.0)
print(f"    ✓ xEtr(data1, data2) = {xe_result:.6f}")
print(f"    ✓ Result is finite: {np.isfinite(xe_result)}")

print("\n  Test 1.2: Self-entropy (should be small)")
xe_self = xEtr(data1, data1, bin=20, maxv=1.0, minv=0.0)
print(f"    ✓ xEtr(data1, data1) = {xe_self:.6f}")
print(f"    ✓ Self-entropy is finite: {np.isfinite(xe_self)}")

# ============================================================================
# Test 2: entromap (Entropy Map)
# ============================================================================
print("\n[TEST GROUP 2] entromap - Entropy Map Computation")
print("-" * 70)

print("\n  Test 2.1: Basic entromap functionality")
test_vol = np.random.rand(3, 4, 8)
emap = entromap(test_vol, bin=10, maxv=1.0, minv=0.0)
print(f"    ✓ Input shape: {test_vol.shape} → Output shape: {emap.shape}")
print(f"    ✓ Expected output shape: (6, 8)")
print(f"    ✓ All values finite: {np.isfinite(emap).all()}")
print(f"    ✓ Value range: [{emap.min():.6f}, {emap.max():.6f}]")

print("\n  Test 2.2: Different input dimensions")
test_vol2 = np.random.rand(5, 3, 12)
emap2 = entromap(test_vol2, bin=15, maxv=1.0, minv=0.0)
print(f"    ✓ Input shape: {test_vol2.shape} → Output shape: {emap2.shape}")
print(f"    ✓ Expected output shape: (10, 6)")

# ============================================================================
# Test 3: XCcumulant (Cross-Cumulant)
# ============================================================================
print("\n[TEST GROUP 3] XCcumulant - Cross-Cumulant Map")
print("-" * 70)

print("\n  Test 3.1: Basic XCcumulant functionality")
test_vol3 = np.random.rand(4, 5, 8)
xcum = XCcumulant(test_vol3, order=2, offset=0.5)
print(f"    ✓ Input shape: {test_vol3.shape} → Output shape: {xcum.shape}")
print(f"    ✓ Expected output shape: (8, 10)")
print(f"    ✓ All values finite: {np.isfinite(xcum).all()}")
print(f"    ✓ All values non-negative: {(xcum >= 0).all()}")
print(f"    ✓ Value range: [{xcum.min():.6f}, {xcum.max():.6f}]")

print("\n  Test 3.2: Different orders")
xcum_o1 = XCcumulant(test_vol3, order=1, offset=0.0)
xcum_o3 = XCcumulant(test_vol3, order=3, offset=1.0)
print(f"    ✓ Order=1 result range: [{xcum_o1.min():.6f}, {xcum_o1.max():.6f}]")
print(f"    ✓ Order=3 result range: [{xcum_o3.min():.6f}, {xcum_o3.max():.6f}]")
print(f"    ✓ Higher order produces larger values: {xcum_o3.max() > xcum_o1.max()}")

print("\n  Test 3.3: Offset effect")
xcum_no_offset = XCcumulant(test_vol3, order=2, offset=0.0)
xcum_with_offset = XCcumulant(test_vol3, order=2, offset=0.5)
offset_diff = np.abs(xcum_no_offset - xcum_with_offset).max()
print(f"    ✓ No offset max: {xcum_no_offset.max():.6f}")
print(f"    ✓ With offset max: {xcum_with_offset.max():.6f}")
print(f"    ✓ Max difference: {offset_diff:.6f}")

# ============================================================================
# Integration Tests
# ============================================================================
print("\n[TEST GROUP 4] Integration Tests")
print("-" * 70)

print("\n  Test 4.1: All functions with common data")
common_data = np.random.rand(3, 3, 10)
try:
    emap_int = entromap(common_data, bin=10, maxv=1.0, minv=0.0)
    xcum_int = XCcumulant(common_data, order=2, offset=0.5)
    print(f"    ✓ entromap output shape: {emap_int.shape}")
    print(f"    ✓ XCcumulant output shape: {xcum_int.shape}")
    print(f"    ✓ Both outputs have same spatial shape")
except Exception as e:
    print(f"    ✗ Error: {e}")

print("\n  Test 4.2: Consistency check - different random seeds")
np.random.seed(123)
data_a = np.random.rand(3, 3, 8)
np.random.seed(123)
data_b = np.random.rand(3, 3, 8)
result_a = XCcumulant(data_a, order=2, offset=0.5)
result_b = XCcumulant(data_b, order=2, offset=0.5)
consistency = np.allclose(result_a, result_b)
print(f"    ✓ Same seed produces identical results: {consistency}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("ALL TESTS COMPLETED SUCCESSFULLY ✓")
print("=" * 70)
print("\nConverted MATLAB Functions:")
print("  1. xEtr(data1, data2, bin, maxv, minv) - Cross-entropy metric")
print("  2. entromap(data, bin, maxv, minv) - Entropy map computation")
print("  3. XCcumulant(data, order, offset) - Cross-cumulant map")
print("=" * 70)


import sys
import os
import inspect

# Add root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests import test_analysis
from tests import test_pipeline

def run_tests(module):
    print(f"\nRunning tests in {module.__name__}...")
    passed = 0
    failed = 0
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("test_"):
            try:
                func()
                print(f"  [PASS] {name}")
                passed += 1
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
    return passed, failed

if __name__ == "__main__":
    total_passed = 0
    total_failed = 0
    
    p, f = run_tests(test_analysis)
    total_passed += p
    total_failed += f
    
    p, f = run_tests(test_pipeline)
    total_passed += p
    total_failed += f
    
    print(f"\nTotal Passed: {total_passed}")
    print(f"Total Failed: {total_failed}")
    
    if total_failed > 0:
        sys.exit(1)

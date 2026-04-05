import sys
print("Running Phase 2 tests manually...")
try:
    from test_phase2 import test_categorization_training_and_prediction, test_spend_model_training_and_prediction
    test_categorization_training_and_prediction()
    print("Categorization Model tests passed!")
    test_spend_model_training_and_prediction()
    print("Expected Spend Model tests passed!")
except Exception as e:
    print(f"Phase 2 Error: {e}")

print("Running Phase 1 tests manually...")
try:
    from test_phase1 import *
    TestNormalizeFlag().test_invalid_strings_return_none("XX")
    print("Phase 1 subset passed!")
except Exception as e:
    print(f"Phase 1 Error: {e}")

import sys
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

# Setup path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.truly_integrated_engine import TrulyIntegratedEngine

def test_o1_attention():
    print("Initializing engine...")
    engine = TrulyIntegratedEngine(str(PROJECT_ROOT / "weights/Qwen3.5-0.8B-Base"))
    engine.initialize()
    
    print("Testing generation (should trigger O(1) slicing)...")
    prompt = "1+1=2, 2+2=4. " * 30  # Make it long enough to exceed O1_WINDOW_SIZE (128)
    
    count = 0
    try:
        for token in engine.generate_stream(prompt, max_new_tokens=20):
            print(token, end="", flush=True)
            count += 1
            if count > 10:
                break
        print("\nSUCCESS: O(1) slicing worked without crashing.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_o1_attention()

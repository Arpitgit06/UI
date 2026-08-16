import subprocess
import sys
import time

def main():
    print("==================================================")
    print("🚀 Starting OmniUI Test Suite")
    print("==================================================")
    
    start_time = time.time()
    
    print("\n[1/2] Running Core & Integration Tests (pytest)...")
    print("This includes all Module A-D unit tests, schema tests, VRAM manager, API smoke, and system stress tests.")
    print("--------------------------------------------------")
    
    import os
    
    # Try to locate the pytest executable inside the local virtual environment
    venv_pytest = os.path.join(".venv", "Scripts", "pytest.exe")
    omniui_pytest = os.path.join("omniui_env", "Scripts", "pytest.exe")
    
    if os.path.exists(venv_pytest):
        pytest_cmd = [venv_pytest, "tests/", "-v", "--disable-warnings"]
    elif os.path.exists(omniui_pytest):
        pytest_cmd = [omniui_pytest, "tests/", "-v", "--disable-warnings"]
    else:
        # Fallback to current environment
        pytest_cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--disable-warnings"]
        
    try:
        # Run and stream output directly to stdout/stderr
        result = subprocess.run(pytest_cmd)
        pytest_success = (result.returncode == 0)
    except FileNotFoundError:
        print("❌ Error: 'pytest' not found. Ensure you have installed requirements-dev.txt")
        pytest_success = False

    end_time = time.time()
    
    print("\n==================================================")
    print("📊 Combined Test Output Summary")
    print("==================================================")
    
    if pytest_success:
        print("✅ Core & Integration Tests: PASSED")
    else:
        print("❌ Core & Integration Tests: FAILED (See logs above for details)")
        
    print(f"\nTotal Testing Time: {end_time - start_time:.2f} seconds")
    
    print("\n> Note: The interactive load test (tests/run_load_test.py) was skipped because it requires")
    print("> an active OmniUI server running (e.g. via ./run.sh or start.bat).")
    print("> To run the heavy load test against a live server, open a new terminal and run:")
    print(">    python -m tests.run_load_test --jobs 10 --complexity high")
    print("==================================================")

    if not pytest_success:
        sys.exit(1)

if __name__ == "__main__":
    main()

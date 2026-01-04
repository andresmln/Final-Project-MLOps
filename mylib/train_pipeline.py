import subprocess
import sys
import time

def run_script(script_name):
    """Runs a python script and captures success/failure."""
    print(f"🚀 Starting {script_name}...")
    start = time.time()
    
    # Run the script as a subprocess
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    
    duration = time.time() - start
    
    if result.returncode == 0:
        print(f"✅ {script_name} finished successfully in {duration:.2f}s.")
        return True
    else:
        print(f"❌ {script_name} FAILED after {duration:.2f}s.")
        return False

if __name__ == "__main__":
    print("==========================================")
    print("      STARTING FULL TRAINING PIPELINE     ")
    print("==========================================")
    
    # 1. Train Champion Model (Critical)
    success_champion = run_script("mylib/train.py")
    
    if not success_champion:
        print("🚨 CRITICAL: Champion model failed to train. Aborting pipeline.")
        sys.exit(1)
        
    # 2. Train Shadow Model (Non-Critical)
    # We run this after the champion. If it fails, we warn but don't break the build.
    success_shadow = run_script("mylib/train_shadow.py")
    
    if not success_shadow:
        print("⚠️ WARNING: Shadow model failed. Deployment will continue with Champion only.")
        # We still exit with 0 (Success) because the main model works.
        sys.exit(0)
    
    print("==========================================")
    print("      PIPELINE COMPLETED SUCCESSFULLY     ")
    print("==========================================")
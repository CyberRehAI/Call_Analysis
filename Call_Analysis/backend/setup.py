#!/usr/bin/env python3
"""
Setup script for Call Analysis System
"""

import subprocess
import sys
import os
from src.call_analysis.demo import DemoSystem

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main setup function"""
    print("Call Analysis System Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    
    # Install dependencies
    print("\nInstalling dependencies...")
    if not run_command("pip install -e .", "Installing package dependencies"):
        print("Failed to install dependencies. Please check your Python environment.")
        sys.exit(1)
    
    # Create output directory
    print("\nCreating output directory...")
    os.makedirs("output", exist_ok=True)
    print("✓ Output directory created")
    
    # Test the demo
    print("\nTesting the demo system...")
    try:
        demo = DemoSystem()
        print("✓ Demo system initialized successfully")
    except Exception as e:
        print(f"✗ Demo system test failed: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 40)
    print("Setup completed successfully!")
    print("\nTo run the demo:")
    print("  python run_demo.py")
    print("\nTo start the web application:")
    print("  python run_web_app.py")
    print("\nTo install additional dependencies manually:")
    print("  pip install -e .")
    print("=" * 40)

if __name__ == "__main__":
    main()

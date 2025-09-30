"""
Simple setup script for the RAG Document Assistant
Run this to set up everything automatically
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "simple_requirements.txt"])
        print("✅ Packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def create_folders():
    """Create necessary folders"""
    print("📁 Creating folders...")
    folders = ["uploads", "templates"]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Created folder: {folder}")

def check_files():
    """Check if all required files exist"""
    print("🔍 Checking files...")
    required_files = [
        "simple_app.py",
        "templates/simple_index.html", 
        "simple_requirements.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files found!")
        return True

def main():
    """Main setup function"""
    print("🚀 Setting up Simple RAG Document Assistant...")
    print("=" * 50)
    
    # Check files
    if not check_files():
        print("❌ Setup failed - missing files")
        return
    
    # Create folders
    create_folders()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed - could not install packages")
        return
    
    print("=" * 50)
    print("✅ Setup complete!")
    print("")
    print("🎉 Ready to start!")
    print("Run: python simple_app.py")
    print("Then open: http://localhost:5000")
    print("")
    print("📚 Read SIMPLE_README.md for more help")

if __name__ == "__main__":
    main()

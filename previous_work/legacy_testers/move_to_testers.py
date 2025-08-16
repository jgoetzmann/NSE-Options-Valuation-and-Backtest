#!/usr/bin/env python3
"""
Legacy Testers - File Mover Script
==================================

This script helps you move legacy test files from the legacy_testers folder
to the main testers folder. Use this if you absolutely need to use a legacy file.

WARNING: Legacy files may be deprecated and may not work with current code!
"""

import os
import shutil
import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("LEGACY TESTERS - FILE MOVER")
    print("=" * 60)
    print()
    
    # Get current directory
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    testers_dir = project_root / "testers"
    
    print(f"Current directory: {current_dir}")
    print(f"Project root: {project_root}")
    print(f"Testers directory: {testers_dir}")
    print()
    
    # Check if testers directory exists
    if not testers_dir.exists():
        print("❌ Error: Main testers directory not found!")
        print(f"Expected location: {testers_dir}")
        return
    
    # List available legacy files
    legacy_files = [f for f in current_dir.glob("*.py") if f.name != "move_to_testers.py"]
    
    if not legacy_files:
        print("ℹ️  No legacy Python files found to move.")
        return
    
    print("📁 Available legacy files:")
    for i, file_path in enumerate(legacy_files, 1):
        print(f"  {i:2d}. {file_path.name}")
    print()
    
    # Ask user which file to move
    try:
        choice = input("Enter the number of the file to move (or 'q' to quit): ").strip()
        if choice.lower() == 'q':
            print("Exiting...")
            return
        
        choice_num = int(choice)
        if choice_num < 1 or choice_num > len(legacy_files):
            print("❌ Invalid choice!")
            return
        
        selected_file = legacy_files[choice_num - 1]
        
    except ValueError:
        print("❌ Please enter a valid number!")
        return
    
    # Confirm the move
    print(f"\n📋 Selected file: {selected_file.name}")
    print(f"📁 Source: {selected_file}")
    print(f"📁 Destination: {testers_dir / selected_file.name}")
    print()
    
    confirm = input("Are you sure you want to move this file? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Move cancelled.")
        return
    
    # Check if destination file already exists
    dest_path = testers_dir / selected_file.name
    if dest_path.exists():
        print(f"\n⚠️  Warning: {dest_path.name} already exists in testers folder!")
        overwrite = input("Do you want to overwrite it? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Move cancelled.")
            return
    
    # Move the file
    try:
        shutil.copy2(selected_file, dest_path)
        print(f"\n✅ Successfully moved {selected_file.name} to testers folder!")
        print(f"📁 Location: {dest_path}")
        print()
        
        # Show next steps
        print("📋 Next steps:")
        print("1. Navigate to the testers folder")
        print("2. Test the file with small datasets")
        print("3. Check for import errors or deprecated functions")
        print("4. Update any broken code before using")
        print()
        print("⚠️  Remember: This is a legacy file and may not work properly!")
        
    except Exception as e:
        print(f"❌ Error moving file: {e}")

if __name__ == "__main__":
    main() 
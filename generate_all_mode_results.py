#!/usr/bin/env python3
"""
Generate All Mode Results
=========================

Simple script to generate comprehensive results for all three execution modes.
This can be run independently or after any mode execution.

Usage:
    python generate_all_mode_results.py
"""

from generate_mode_results import EnhancedModeResultsGenerator

def main():
    print("Generating comprehensive results for all execution modes...")
    print("=" * 60)
    
    generator = EnhancedModeResultsGenerator()
    
    # Generate individual mode reports
    print("\n1. Generating Mode A Results...")
    mode_a_report = generator.generate_mode_a_results()
    
    print("\n2. Generating Mode B Results...")
    mode_b_report = generator.generate_mode_b_results()
    
    print("\n3. Generating Mode C Results...")
    mode_c_report = generator.generate_mode_c_results()
    
    # Generate comprehensive summary
    print("\n4. Generating Comprehensive Summary...")
    comprehensive_report = generator.generate_comprehensive_summary()
    
    print("\n" + "=" * 60)
    print("ALL MODE RESULTS GENERATED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\nGenerated Reports:")
    print(f"  📊 Mode A Results: {mode_a_report}")
    print(f"  🤖 Mode B Results: {mode_b_report}")
    print(f"  📈 Mode C Results: {mode_c_report}")
    print(f"  🔍 Comprehensive Summary: {comprehensive_report}")
    
    print(f"\nAll reports saved to: {generator.output_dir}")
    
    print("\nNext Steps:")
    print("  1. Review individual mode reports for detailed insights")
    print("  2. Check comprehensive summary for system-wide analysis")
    print("  3. Use insights to optimize trading strategies")
    print("  4. Monitor system health and performance metrics")
    
    print("\nNote: These reports are automatically generated after each mode execution.")
    print("You can also run this script independently anytime to refresh the analysis.")

if __name__ == "__main__":
    main()

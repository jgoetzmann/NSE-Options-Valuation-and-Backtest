#!/usr/bin/env python3
"""
Mode B Runner: Train-and-Score ML on Historical Reconstructions
==============================================================
Simple script to run Mode B and show where to find results.
"""

import os
import sys
import glob
from datetime import datetime

def main():
    print("🤖 MODE B: Train-and-Score ML on Historical Reconstructions")
    print("=" * 70)
    
    # Check if required modules exist
    if not os.path.exists("models/train.py"):
        print("❌ Training module not found!")
        print("   Make sure you're in the project root directory")
        return
    
    if not os.path.exists("models/score_snapshot.py"):
        print("❌ Scoring module not found!")
        print("   Make sure you're in the project root directory")
        return
    
    # Check for config
    config_path = "configs/ml_experiment.yml"
    if not os.path.exists(config_path):
        print(f"❌ ML config file not found: {config_path}")
        print("   Please create the config file first")
        return
    
    print("📋 ML Configuration found!")
    
    # Step 1: Training
    print("\n🔄 STEP 1: Training ML Models...")
    print("   This may take 10-30 minutes depending on data size...")
    
    try:
        # Import and run training
        sys.path.append("models")
        from train import MLModelTrainer
        
        trainer = MLModelTrainer(config_path)
        training_results = trainer.run_training()
        
        print("\n✅ TRAINING COMPLETED SUCCESSFULLY!")
        
        # Show training results
        if 'model_paths' in training_results:
            print("\n📁 TRAINED MODELS SAVED TO:")
            for model_type, model_path in training_results['model_paths'].items():
                print(f"   {model_type}: {model_path}")
        
        if 'evaluation_results' in training_results:
            print("\n📊 TRAINING PERFORMANCE:")
            eval_results = training_results['evaluation_results']
            if 'classifier' in eval_results:
                cls_metrics = eval_results['classifier']
                print(f"   Classifier ROC-AUC: {cls_metrics.get('roc_auc', 'N/A'):.3f}")
                print(f"   Classifier PR-AUC: {cls_metrics.get('pr_auc', 'N/A'):.3f}")
            
            if 'regressor' in eval_results:
                reg_metrics = eval_results['regressor']
                print(f"   Regressor RMSE: {reg_metrics.get('rmse', 'N/A'):.3f}")
                print(f"   Regressor MAE: {reg_metrics.get('mae', 'N/A'):.3f}")
        
    except Exception as e:
        print(f"\n❌ ERROR during training: {str(e)}")
        print("\n🔧 TROUBLESHOOTING:")
        print("   1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("   2. Check that you have historical data in reconstructed/parquet/")
        print("   3. Verify the ML config file is valid")
        return
    
    # Step 2: Scoring Current Snapshot
    print("\n🔄 STEP 2: Scoring Current Snapshot...")
    
    # Find the most recent snapshot
    snapshot_files = glob.glob("outputs/json/option_chain_slim_NIFTY_*.json")
    if not snapshot_files:
        print("❌ No snapshot files found!")
        print("   Run: python nse_options_scraper.py")
        return
    
    latest_snapshot = max(snapshot_files, key=os.path.getctime)
    print(f"📁 Using snapshot: {os.path.basename(latest_snapshot)}")
    
    try:
        # Import and run scoring
        from score_snapshot import SnapshotScorer
        
        scorer = SnapshotScorer()
        scoring_results = scorer.score_snapshot(latest_snapshot)
        
        print("\n✅ SCORING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        # Show where to find results
        print("\n📊 RESULTS LOCATIONS:")
        print("   📁 Scored Snapshot CSV:")
        csv_files = glob.glob("outputs/csv/scored_snapshot_*.csv")
        for csv_file in csv_files:
            print(f"      {csv_file}")
        
        print("\n   📄 Scoring Report:")
        report_files = glob.glob("outputs/results_from_nse_valuations/scoring_report_*.txt")
        for report_file in report_files:
            print(f"      {report_file}")
        
        print("\n   📈 Ranking Report:")
        ranking_files = glob.glob("outputs/results_from_nse_valuations/ranking_report_*.txt")
        for ranking_file in ranking_files:
            print(f"      {ranking_file}")
        
        print("\n🎯 TOP OPPORTUNITIES:")
        if 'ranked_opportunities' in scoring_results:
            ranked_df = scoring_results['ranked_opportunities']
            if not ranked_df.empty:
                print("   Top 5 ranked options:")
                for i, (_, row) in enumerate(ranked_df.head().iterrows()):
                    print(f"      {i+1}. {row['symbol']} {row['strike']} {row['option_type']} - Score: {row.get('model_score', 'N/A'):.3f}")
        
        print("\n💡 NEXT STEPS:")
        print("   1. Check the scored CSV for ML predictions on each option")
        print("   2. Review the ranking report for top opportunities")
        print("   3. Use the model scores to make trading decisions")
        print("   4. Retrain models periodically with new historical data")
        
        # Generate enhanced results automatically
        print("\n📊 GENERATING ENHANCED RESULTS...")
        try:
            from generate_mode_results import EnhancedModeResultsGenerator
            generator = EnhancedModeResultsGenerator()
            mode_b_report = generator.generate_mode_b_results()
            print(f"✅ Enhanced results generated: {os.path.basename(mode_b_report)}")
        except Exception as e:
            print(f"⚠️  Could not generate enhanced results: {e}")
        
    except Exception as e:
        print(f"\n❌ ERROR during scoring: {str(e)}")
        print("\n🔧 TROUBLESHOOTING:")
        print("   1. Make sure training completed successfully")
        print("   2. Check that models are saved in models/model_store/")
        print("   3. Verify the snapshot file is valid")

if __name__ == "__main__":
    main()

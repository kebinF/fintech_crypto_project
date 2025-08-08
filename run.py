"""
Simple Main Program for Cryptocurrency Portfolio Management System
Runs Stage 1-4 in sequence
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        'data/raw',
        'data/processed',
        'data/cache',
        'data/features',
        'data/results',
        'reports',
        'logs'
    ]
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    logger.info("All directories verified/created")

def run_stage1():
    """Stage 1: Data Collection"""
    logger.info("="*50)
    logger.info("Starting Stage 1: Data Collection")
    logger.info("="*50)

    try:
        from pipeline import stage1_data_collection

        # Run the main data collection function if it exists
        if hasattr(stage1_data_collection, 'main'):
            stage1_data_collection.main()
        elif hasattr(stage1_data_collection, 'stage1_data_collection'):
            # Call the stage1 function with default parameters
            api_key = "340031ae9d60a76b565ef5473187110a1982bfeb99bc1b6ee73545f7aa694446"
            data = stage1_data_collection.stage1_data_collection(
                api_key=api_key,
                pages=[1, 2],  # Get top 200 coins
                data_dir=Path('data/processed')
            )
            logger.info(f"Stage 1 completed: {len(data)} records collected")
        else:
            logger.warning("Stage 1: No main function found, trying to import and run available functions")
            # Try to run any available collection function
            if hasattr(stage1_data_collection, 'get_daily_ohlcv'):
                logger.info("Running basic data collection...")
            else:
                logger.error("No suitable function found in stage1_data_collection")

        logger.info("Stage 1: Data Collection Completed")

    except Exception as e:
        logger.error(f"Stage 1 failed: {e}")
        logger.info("Continuing to next stage...")

def run_stage2():
    """Stage 2: Feature Engineering"""
    logger.info("="*50)
    logger.info("Starting Stage 2: Feature Engineering")
    logger.info("="*50)

    try:
        from pipeline import stage2_feature_engineering

        # Check for different possible function names
        if hasattr(stage2_feature_engineering, 'main'):
            stage2_feature_engineering.main()
        elif hasattr(stage2_feature_engineering, 'stage2_feature_engineering'):
            # Load data from stage 1
            import pandas as pd
            input_file = Path('data/processed/stage_1_crypto_data.csv')

            if input_file.exists():
                data = pd.read_csv(input_file, index_col=['symbol', 'date'], parse_dates=['date'])
                features = stage2_feature_engineering.stage2_feature_engineering(
                    tidy_prices=data,
                    data_dir=Path('data/processed')
                )
                logger.info(f"Stage 2 completed: {features.shape} features generated")
            else:
                logger.warning("No input data found from Stage 1")
        else:
            logger.warning("Stage 2: No suitable function found")

        logger.info("Stage 2: Feature Engineering Completed")

    except Exception as e:
        logger.error(f"Stage 2 failed: {e}")
        logger.info("Continuing to next stage...")

def run_stage3():
    """Stage 3: Model Design & Portfolio Optimization"""
    logger.info("="*50)
    logger.info("Starting Stage 3: Model Design")
    logger.info("="*50)

    try:
        from pipeline import stage3_model_design

        if hasattr(stage3_model_design, 'main'):
            stage3_model_design.main()
        elif hasattr(stage3_model_design, 'optimize_portfolio'):
            # Load features from stage 2
            import pandas as pd
            features_file = Path('data/processed/stage_2_crypto_data.csv')

            if features_file.exists():
                features = pd.read_csv(features_file)
                weights = stage3_model_design.optimize_portfolio(features)
                logger.info(f"Stage 3 completed: Portfolio optimized")
            else:
                logger.warning("No features found from Stage 2")
        else:
            logger.warning("Stage 3: No suitable function found")

        logger.info("Stage 3: Model Design Completed")

    except Exception as e:
        logger.error(f"Stage 3 failed: {e}")
        logger.info("Continuing to next stage...")

def run_stage4():
    """Stage 4: Application Implementation (Dashboard)"""
    logger.info("="*50)
    logger.info("Starting Stage 4: Application Implementation")
    logger.info("="*50)

    try:
        from pipeline import stage4_app_implementation

        # Check if it's a Streamlit app
        if hasattr(stage4_app_implementation, 'main'):
            logger.info("Running dashboard main function...")
            stage4_app_implementation.main()
        else:
            # Try to run as Streamlit app
            logger.info("Launching Streamlit dashboard...")
            logger.info("Please open your browser to http://localhost:8501")

            import subprocess
            import sys

            # Get the path to stage4 file
            stage4_path = Path('src/pipeline/stage4_app_implementation.py')

            if stage4_path.exists():
                # Run streamlit
                subprocess.run([sys.executable, '-m', 'streamlit', 'run', str(stage4_path)])
            else:
                logger.error(f"Could not find {stage4_path}")
                logger.info("Please run manually: streamlit run src/pipeline/stage4_app_implementation.py")

    except Exception as e:
        logger.error(f"Stage 4 failed: {e}")
        logger.info("You may need to run the dashboard manually")

def main():
    """Main entry point - runs all 4 stages in sequence"""

    print(""" Cryptocurrency Portfolio Management System     
                  Running Pipeline Stages 1-4     
    """)

    # Ensure all directories exist
    ensure_directories()

    # Ask user what to run
    print("\nSelect execution mode:")
    print("1. Run all stages (1-4)")
    print("2. Run Stage 1 only (Data Collection)")
    print("3. Run Stage 2 only (Feature Engineering)")
    print("4. Run Stage 3 only (Model Design)")
    print("5. Run Stage 4 only (Dashboard)")
    print("6. Run stages 1-3 (without Dashboard)")

    choice = input("\nEnter your choice (1-6): ").strip()

    if choice == '1':
        # Run all stages
        run_stage1()
        run_stage2()
        run_stage3()
        run_stage4()
    elif choice == '2':
        run_stage1()
    elif choice == '3':
        run_stage2()
    elif choice == '4':
        run_stage3()
    elif choice == '5':
        run_stage4()
    elif choice == '6':
        run_stage1()
        run_stage2()
        run_stage3()
        logger.info("Stages 1-3 completed. Dashboard not launched.")
    else:
        logger.error("Invalid choice. Please run again and select 1-6.")
        return

    logger.info("="*50)
    logger.info("Pipeline execution completed!")
    logger.info("="*50)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nExecution interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
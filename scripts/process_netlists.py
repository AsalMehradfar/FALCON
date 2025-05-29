import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

from data_modules.netlist2graph import process_all_netlists


if __name__ == "__main__":
    # Run the processing
    root_dir = "dataset"  # Change this if your dataset is stored elsewhere
    process_all_netlists(root_dir)
import os
import subprocess
import pytest

def test_run_tracking_creates_detection_results(test_case_dir):
    """
    Test that running main.py with the test config always creates a new detection_results.nc file.
    """
    case_dir = test_case_dir['case_dir']
    config_file = test_case_dir['config_file']

    # The detection_results.nc file should always be freshly created.
    # If it exists, remove it before running main.py
    detection_results_file = os.path.join(case_dir, 'detection_results.nc')
    if os.path.exists(detection_results_file):
        os.remove(detection_results_file)

    # Run main.py with the provided config
    result = subprocess.run(['python', './code/main.py', '--config', config_file],
                            capture_output=True, text=True)
    assert result.returncode == 0, f"main.py failed for {test_case_dir['case_name']}:\n{result.stderr}"

    # Now detection_results.nc should be created again
    assert os.path.exists(detection_results_file), "detection_results.nc was not created."

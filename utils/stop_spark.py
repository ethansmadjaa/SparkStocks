import os
import signal
import subprocess
import sys
from typing import List


def get_spark_process_ids() -> List[int]:
    """Find all Spark-related process IDs."""
    pids = []
    current_pid = os.getpid()
    
    try:
        # Different commands for different operating systems
        if sys.platform == "win32":
            cmd = ["tasklist", "/FI", "IMAGENAME eq java.exe", "/FO", "CSV"]
        else:
            cmd = ["ps", "aux"]
            
        output = subprocess.check_output(cmd).decode()
        
        # Look for java processes running Spark
        for line in output.split('\n'):
            if 'spark' in line.lower():
                # Extract PID based on OS
                if sys.platform == "win32":
                    if "java.exe" in line:
                        pid = int(line.split(',')[1].strip('"'))
                        if pid != current_pid:  # Don't include our own process
                            pids.append(pid)
                else:
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            pid = int(parts[1])
                            if pid != current_pid:  # Don't include our own process
                                pids.append(pid)
                        except ValueError:
                            continue
                            
    except subprocess.CalledProcessError:
        print("Error getting process list")
        
    return pids


def stop_spark():
    """Stop all Spark sessions and clean up resources."""
    # Get Spark process IDs
    pids = get_spark_process_ids()
    
    # Kill each process
    for pid in pids:
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/PID", str(pid)])
            else:
                os.kill(pid, signal.SIGKILL)
            print(f"Terminated Spark process {pid}")
        except (ProcessLookupError, PermissionError) as e:
            print(f"Could not terminate process {pid}: {e}")
    
    # Clean up temporary directories
    try:
        if sys.platform == "win32":
            os.system("rd /s /q %tmp%\\spark-*")
        else:
            os.system("rm -rf /tmp/spark-*")
        print("Cleaned up temporary Spark directories")
    except Exception as e:
        print(f"Error cleaning temporary directories: {e}")


if __name__ == "__main__":
    stop_spark()
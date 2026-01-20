"""
Entry point for Auto-Blogging System.
Delegates to src.main.
"""
import sys
import os

# Add the current directory to python path
sys.path.append(os.getcwd())

from src.main import initialize_system, run_content_generation, run_maintenance

if __name__ == "__main__":
    system = initialize_system()
    if system:
        if len(sys.argv) > 1:
            command = sys.argv[1]
            
            if command == "maintenance":
                limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
                run_maintenance(system, limit=limit)
            elif command == "weekly":
                run_content_generation(system, mode="weekly")
            elif command == "daily":
                run_content_generation(system, mode="daily")
            else:
                # Assume manual topic
                run_content_generation(system, mode="manual", manual_topic=command)
        else:
            # Default behavior (Daily)
            run_content_generation(system, mode="daily")

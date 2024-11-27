from security_system import SecuritySystem
from utils import setup_logging

def main():
    # Setup logging
    logger = setup_logging()

    # Initialize the security system
    security_system = SecuritySystem()

    logger.info("Starting the security system...")
    security_system.start_monitoring()

if __name__ == "__main__":
    main()

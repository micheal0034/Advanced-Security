import logging

def setup_logging():
    """Setup basic logging for the security system"""
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('SecuritySystem')
    return logger

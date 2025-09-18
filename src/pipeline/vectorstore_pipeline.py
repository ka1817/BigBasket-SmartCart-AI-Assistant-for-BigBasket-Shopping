import logging
from src.vectorstore import VectorStoreManager

logger = logging.getLogger("VectorStorePipeline")

class VectorStorePipeline:
    def run(self) -> VectorStoreManager:
        logger.info(">>> Running Vector Store Pipeline (load only)...")
        vs_manager = VectorStoreManager()
        vs_manager.create_or_load_index()
        logger.info("Vector store loaded successfully.")
        return vs_manager


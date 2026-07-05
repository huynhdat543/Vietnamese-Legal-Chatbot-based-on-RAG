from qdrant_client import QdrantClient
from config.settings import settings

class QdrantManager:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )

    def get_client(self) -> QdrantClient:
        return self.client

    def check_connection(self):
        try:
            self.client.get_collection(collection_name=settings.COLLECTION_NAME)
            print(f"Kết nối Qdrant thành công: {settings.COLLECTION_NAME}")
            return True
        except Exception as e:
            print(f"Lỗi kết nối: {e}")
            return False
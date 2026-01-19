from typing import List
from langchain_core.embeddings import Embeddings

class ZhipuAIEmbeddings(Embeddings):
    def __init__(self, batch_size: int = 64):
        from zhipuai import ZhipuAI

        self.batch_size = batch_size
        self.client = ZhipuAI(
            api_key='b9ca069cc7e144a6939f0c17f3db141f.27Dpwg8sKLCDcNT7'
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            resp = self.client.embeddings.create(
                model="embedding-3",
                input=batch_texts
            )

            batch_embeddings = [
                item.embedding for item in resp.data
            ]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

from ast import mod
import json

import requests

DIAL_EMBEDDINGS = "https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings"


# TODO:
# ---
# https://dialx.ai/dial_api#operation/sendEmbeddingsRequest
# ---
# Implement DialEmbeddingsClient:
# - constructor should apply deployment name and api key
# - create method `get_embeddings` that will generate embeddings for input list (don't forget about dimensions)
#   with Embedding model and return back a dict with indexed embeddings (key is index from input list and value vector list)


class DialEmbeddingsClient:
    _endpoint: str
    _apikey: str

    def __init__(self, deployment_name: str, api_key: str):
        if not api_key or api_key.strip() == "":
            raise ValueError("API key cannot be null or empty")
        self._endpoint = DIAL_EMBEDDINGS.format(model=deployment_name)
        self._api_key = api_key

    def get_embeddings(self, text: list[str], dimensions: int) -> dict:
        """
        Generate embedding from a given text

        Args:
            input: can be a text or list of texts
            dimensions: number of dimensions
        """

        print("Getting embeddings\n")

        headers = {"api-key": self._api_key, "Content-Type": "application/json"}
        request_data = {input: text, dimensions: dimensions}

        response = requests.post(
            url=self._endpoint, headers=headers, json=request_data, timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            print("\n", "=" * 50, "\n", "RESPONSE\n", data, "\n", "=" * 50)
            return self._from_data(data)
        else:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

    def _from_data(self, data) -> dict[int, list[float]]:
        return {
            embedding["index"]: embedding["embedding"] for embedding in data["data"]
        }


# Hint:
#  Response JSON:
#  {
#     "data": [
#         {
#             "embedding": [
#                 0.19686688482761383,
#                 ...
#             ],
#             "index": 0,
#             "object": "embedding"
#         }
#     ],
#     ...
#  }

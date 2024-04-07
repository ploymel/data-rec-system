from sentence_transformers import SentenceTransformer
import torch
from sentence_transformers import util
from transformers import utils
import pandas as pd
from openai import OpenAI

utils.logging.set_verbosity_error()


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer("weight/model-sbert")
        data_collection = pd.read_csv("dataset_collection.csv")
        self.data_collection = data_collection.dropna(subset="description")
        self.collection_embeddings = self.model.encode(
            self.data_collection["description"].to_list(), convert_to_tensor=True
        )

    def retrieve_candidates(self, query, top_k, use_reranker=False):
        if use_reranker:
            client = OpenAI()
            top_k = top_k * 2

        print(top_k)

        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.collection_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        if use_reranker:
            input_text = "Query:\n"
            input_text = input_text + query + "\n"

            input_text = input_text + "Candidate Dataset:" + "\n"
            k = 0

            output = dict()
            for score, idx in zip(top_results[0], top_results[1]):
                record = self.data_collection.iloc[[idx.cpu().item()]].to_dict(
                    orient="records"
                )[0]
                input_text = input_text + str(k) + ". " + record["name"] + "\n"
                input_text = input_text + "description: " + record["description"] + "\n"

                output[k] = {"idx": idx, "score": score}

                k += 1

            input_text = (
                input_text
                + "Please Rank the most relevant dataset to the given query from the most relevant to less relevant."
                + "\n"
            )
            input_text = input_text + "Return index of dataset only." + "\n"
            input_text = input_text + "For example" + "\n"
            input_text = input_text + "Output: 3, 2, 4, 6, 7, 0, 1, 9, 8, 5" + "\n"

            completion = client.chat.completions.create(
                model="gpt-4", messages=[{"role": "user", "content": input_text}]
            )

            try:
                o = (
                    completion.choices[0]
                    .message.content.lower()
                    .replace("output:", "")
                    .strip()
                )
                new_idx = [int(o_i) for o_i in o.split(", ")]
                final_outputs = []
                for n in new_idx[: top_k // 2]:
                    final_outputs.append(
                        self.data_collection.iloc[[output[n]["idx"].item()]]
                    )

                return final_outputs
            except:
                print("Re-ranker is not working!!")
                output = list()
                for _, idx in zip(top_results[0], top_results[1]):
                    output.append(self.data_collection.iloc[[idx.cpu().item()]])
                return output

        else:
            output = list()
            for _, idx in zip(top_results[0], top_results[1]):
                output.append(self.data_collection.iloc[[idx.cpu().item()]])

            return output


from pathlib import Path
from typing import Any

from src.retriever import PolicyRetriever


class PolicyRAG:
    """Policy question answering with evidence filtering and citations."""

    def __init__(
        self,
        storage_folder: str | Path = "storage",
        min_similarity: float = 0.35,
    ) -> None:
        self.retriever = PolicyRetriever()
        self.retriever.load_index(storage_folder)
        self.min_similarity = min_similarity

    def ask(self, question: str, top_k: int = 3) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("Please enter a question.")

        results = self.retriever.search(question, top_k=top_k)

        evidence = [
            item for item in results
            if item["similarity"] >= self.min_similarity
        ]

        if not evidence:
            return {
                "question": question,
                "answer": (
                    "I couldn't find sufficiently relevant evidence in "
                    "the available policy documents. Please consult your "
                    "manager or Human Resources."
                ),
                "sources": [],
                "supported": False,
            }

        sources = [
            {
                "document": item["source"],
                "page": item["page"],
                "similarity": round(item["similarity"], 3),
                "evidence": item["text"],
            }
            for item in evidence
        ]

        answer = (
            "Based on the available policy documents, the relevant "
            "policy text is provided below:\n\n"
            + "\n\n".join(item["text"] for item in evidence)
        )

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "supported": True,
        }


if __name__ == "__main__":
    pipeline = PolicyRAG()

    test_questions = [
        "What should I do if I arrive late?",
        "Who must approve working remotely?",
        "What is the company's maternity leave allowance?",
    ]

    for question in test_questions:
        result = pipeline.ask(question)

        print("\n" + "=" * 60)
        print("QUESTION:", result["question"])
        print("SUPPORTED:", result["supported"])
        print("\nANSWER:\n", result["answer"])
        print("\nSOURCES:")

        if not result["sources"]:
            print("No sufficiently relevant source found.")
        else:
            for source in result["sources"]:
                print(
                    f"- {source['document']}, page {source['page']} "
                    f"(similarity: {source['similarity']})"
                )

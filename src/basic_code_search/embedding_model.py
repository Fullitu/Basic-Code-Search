from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

class EmbeddingModel:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def get_model(self) -> SentenceTransformer:
        return self.model

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts)

    def train(self, train_dataset, eval_dataset, loss_function, args: SentenceTransformerTrainingArguments):
        loss = loss_function(self.model)
        trainer = SentenceTransformerTrainer(
            self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            args=args,
        )
        trainer.train()
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from transformers import TrainerCallback

class EmbeddingModel:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def get_model(self) -> SentenceTransformer:
        return self.model

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts)

    def train(self, train_dataset, eval_dataset, loss_function, args: SentenceTransformerTrainingArguments):
        train_loss_history = []
        eval_loss_history = []

        class StateSaveCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                train_loss = logs.pop("loss", None)
                eval_loss = logs.pop("eval_loss", None)
                
                if train_loss is not None:
                    train_loss_history.append(train_loss)
                if eval_loss is not None:
                    eval_loss_history.append(eval_loss)


        loss = loss_function(self.model)
        trainer = SentenceTransformerTrainer(
            self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            args=args,
            callbacks=[StateSaveCallback]
        )
        trainer.train()

        return {
            "train_loss_history": train_loss_history,
            "eval_loss_history": eval_loss_history
        }
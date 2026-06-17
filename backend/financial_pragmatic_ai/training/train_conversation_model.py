import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader

from financial_pragmatic_ai.analysis.conversation_vectorizer import vectorize_conversation
from financial_pragmatic_ai.models.conversation_interaction_model import ConversationInteractionModel


DATA_PATH = "financial_pragmatic_ai/data/conversation_signal_dataset.csv"

SIGNAL_INDEX = {
    "neutral": 0,
    "risk": 1,
    "growth": 2
}


class ConversationDataset(Dataset):

    def __init__(self, df):

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        intents = [
            {"speaker": "CEO", "intent": row["CEO_intent"]},
            {"speaker": "CFO", "intent": row["CFO_intent"]},
            {"speaker": "ANALYST", "intent": row["Analyst_intent"]}
        ]

        vec = vectorize_conversation(intents)

        label = SIGNAL_INDEX[row["signal"]]

        return vec.float(), torch.tensor(label)


def main():

    df = pd.read_csv(DATA_PATH)

    dataset = ConversationDataset(df)

    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = ConversationInteractionModel()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.CrossEntropyLoss()

    epochs = 200

    for epoch in range(epochs):

        total_loss = 0

        for x, y in loader:

            logits = model(x)

            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 20 == 0:

            print(f"Epoch {epoch} Loss {total_loss:.4f}")

    torch.save(
        model.state_dict(),
        "financial_pragmatic_ai/models/conversation_signal_model.pt"
    )

    print("Model saved")


if __name__ == "__main__":
    main()
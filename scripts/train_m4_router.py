import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# PATHS
# ============================================================

M4_DIR = (
    r"D:\DLPROJECT\data\processed\multimodal"
    r"\m4"
)

TRAIN_FILE = os.path.join(
    M4_DIR,
    "training_pairs.npz"
)

MODEL_FILE = os.path.join(
    M4_DIR,
    "adaptive_router.pt"
)


# ============================================================
# SETTINGS
# ============================================================

EMBED_DIM = 768

HIDDEN_DIM = 128

BATCH_SIZE = 64

EPOCHS = 20

LEARNING_RATE = 1e-3

MARGIN = 0.10

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("M4: ADAPTIVE TEXT + IMAGE ROUTER")
print("=" * 70)

print("\nDevice:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading training pairs...")

data = np.load(
    TRAIN_FILE,
    allow_pickle=True
)

queries = data["queries"].astype(
    np.float32
)

positive_text = data[
    "positive_text"
].astype(np.float32)

positive_image = data[
    "positive_image"
].astype(np.float32)

negative_text = data[
    "negative_text"
].astype(np.float32)

negative_image = data[
    "negative_image"
].astype(np.float32)


print(
    "Training pairs:",
    len(queries)
)

print(
    "Query shape:",
    queries.shape
)


# ============================================================
# CONVERT TO TORCH
# ============================================================

queries = torch.from_numpy(
    queries
)

positive_text = torch.from_numpy(
    positive_text
)

positive_image = torch.from_numpy(
    positive_image
)

negative_text = torch.from_numpy(
    negative_text
)

negative_image = torch.from_numpy(
    negative_image
)


dataset = TensorDataset(
    queries,
    positive_text,
    positive_image,
    negative_text,
    negative_image
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# ============================================================
# ROUTER
# ============================================================

class AdaptiveRouter(nn.Module):

    def __init__(
        self,
        input_dim=768,
        hidden_dim=128
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                2
            )
        )


    def forward(self, query):

        logits = self.network(
            query
        )

        weights = torch.softmax(
            logits,
            dim=1
        )

        return weights


# ============================================================
# MODEL
# ============================================================

model = AdaptiveRouter(
    EMBED_DIM,
    HIDDEN_DIM
).to(device)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\nTraining router...")

best_loss = float("inf")


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    epoch_loss = 0.0

    samples = 0


    for (
        query,
        pos_text,
        pos_image,
        neg_text,
        neg_image
    ) in loader:

        query = query.to(device)

        pos_text = pos_text.to(device)
        pos_image = pos_image.to(device)

        neg_text = neg_text.to(device)
        neg_image = neg_image.to(device)


        # ----------------------------------------------------
        # Query-dependent weights
        # ----------------------------------------------------

        weights = model(
            query
        )

        alpha_text = weights[:, 0:1]

        alpha_image = weights[:, 1:2]


        # ----------------------------------------------------
        # Positive fused representation
        # ----------------------------------------------------

        positive_fused = (
            alpha_text * pos_text
            +
            alpha_image * pos_image
        )


        # ----------------------------------------------------
        # Negative fused representation
        # ----------------------------------------------------

        negative_fused = (
            alpha_text * neg_text
            +
            alpha_image * neg_image
        )


        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        positive_fused = torch.nn.functional.normalize(
            positive_fused,
            dim=1
        )

        negative_fused = torch.nn.functional.normalize(
            negative_fused,
            dim=1
        )


        # ----------------------------------------------------
        # Similarity
        # ----------------------------------------------------

        positive_score = torch.sum(
            query * positive_fused,
            dim=1
        )

        negative_score = torch.sum(
            query * negative_fused,
            dim=1
        )


        # ----------------------------------------------------
        # Margin ranking loss
        # ----------------------------------------------------

        target = torch.ones_like(
            positive_score
        )

        loss = torch.nn.functional.margin_ranking_loss(
            positive_score,
            negative_score,
            target,
            margin=MARGIN
        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        epoch_loss += (
            loss.item()
            * query.size(0)
        )

        samples += query.size(0)


    epoch_loss /= samples


    print(
        f"Epoch {epoch:02d}/{EPOCHS} "
        f"- Loss: {epoch_loss:.6f}"
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if epoch_loss < best_loss:

        best_loss = epoch_loss

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "input_dim":
                    EMBED_DIM,

                "hidden_dim":
                    HIDDEN_DIM,

                "best_loss":
                    best_loss
            },
            MODEL_FILE
        )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("M4 ROUTER TRAINING COMPLETE")
print("=" * 70)

print(
    "Best training loss:",
    best_loss
)

print(
    "Model saved:",
    MODEL_FILE
)

print("=" * 70)
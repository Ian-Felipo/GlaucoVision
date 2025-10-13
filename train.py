import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from tqdm import tqdm
from data_preparation import load_dataset, create_kfold_loaders
from models import load_transformer_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_fold(model, train_loader, val_loader, optimizer, criterion, epochs=10):
    """
    Executa o treinamento e validação para um único fold.
    """
    model.to(DEVICE)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Treinando (Época {epoch+1}/{epochs})", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        model.eval()
        val_loss = 0.0
        preds, probs, gts = [], [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs).logits
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                pred = torch.argmax(outputs, dim=1)
                preds.extend(pred.cpu().numpy())
                probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                gts.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        history["val_loss"].append(val_loss)

        acc = accuracy_score(gts, preds)
        prec = precision_score(gts, preds)
        rec = recall_score(gts, preds)
        f1 = f1_score(gts, preds)
        auc = roc_auc_score(gts, probs)

        print(f"📘 Época {epoch+1}/{epochs} | Loss treino: {train_loss:.4f} | Loss val: {val_loss:.4f}")
        print(f"➡️  ACC={acc*100:.2f}% | PREC={prec*100:.2f}% | REC={rec*100:.2f}% | F1={f1*100:.2f}% | AUC={auc*100:.2f}%")

    return model, (acc, prec, rec, f1, auc)


def kfold_training(data_root="data", model_name="vit", k_fold=5, epochs=10, lr=1e-4, batch_size=32):
    """
    Executa o treinamento com validação cruzada K-fold,
    mostrando métricas por fold e resultados médios finais.
    """
    print(f"\n🚀 Iniciando K-Fold Training ({k_fold} folds) — Modelo: {model_name.upper()}\n")
    dataset = load_dataset(data_root, model_name)
    folds = create_kfold_loaders(dataset, k_fold=k_fold, batch_size=batch_size)

    all_metrics = []

    for fold_idx, (train_loader, val_loader) in enumerate(folds, start=1):
        print(f"\n==============================")
        print(f"🔹 FOLD {fold_idx}/{k_fold} — Treinando modelo {model_name.upper()}")
        print(f"==============================")

        _, model = load_transformer_model(model_name, num_classes=2)
        model.to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        trained_model, metrics = train_one_fold(model, train_loader, val_loader, optimizer, criterion, epochs)
        all_metrics.append(metrics)

        acc, prec, rec, f1, auc = metrics
        print(f"\n📊 Resultados do Fold {fold_idx}:")
        print(f"  Accuracy:     {acc*100:.2f}%")
        print(f"  Precision:    {prec*100:.2f}%")
        print(f"  Recall:       {rec*100:.2f}%")
        print(f"  F1-Score:     {f1*100:.2f}%")
        print(f"  AUC:          {auc*100:.2f}%")

    all_metrics = np.array(all_metrics)
    mean_metrics = all_metrics.mean(axis=0)
    std_metrics = all_metrics.std(axis=0)

    print(f"\n✅ MÉDIA FINAL ({model_name.upper()} — {k_fold} folds):")
    print(f"  Accuracy:     {mean_metrics[0]*100:.2f}% ± {std_metrics[0]*100:.2f}%")
    print(f"  Precision:    {mean_metrics[1]*100:.2f}% ± {std_metrics[1]*100:.2f}%")
    print(f"  Recall:       {mean_metrics[2]*100:.2f}% ± {std_metrics[2]*100:.2f}%")
    print(f"  F1-Score:     {mean_metrics[3]*100:.2f}% ± {std_metrics[3]*100:.2f}%")
    print(f"  AUC:          {mean_metrics[4]*100:.2f}% ± {std_metrics[4]*100:.2f}%")

    return all_metrics


if __name__ == "__main__":
    kfold_training(
        data_root="data",
        model_name="vit",
        k_fold=5,
        epochs=10,
        lr=1e-4,
        batch_size=16
    )

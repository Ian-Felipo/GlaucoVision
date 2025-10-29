import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from tqdm import tqdm
import os

from data_preparation import load_dataset, create_kfold_loaders
from model import load_transformer_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_fold(model, processor, train_loader, val_loader, optimizer, criterion, model_name, fold_idx, epochs=10, save_dir="results"):
    """
    Executa o treinamento e validação para um único fold e salva as probabilidades.
    
    Parâmetros:
        model: modelo Transformer
        processor: AutoImageProcessor do modelo (para normalização correta)
        train_loader: DataLoader de treino
        val_loader: DataLoader de validação
        optimizer: otimizador
        criterion: função de perda
        model_name: nome do modelo
        fold_idx: índice do fold atual
        epochs: número de épocas
        save_dir: diretório para salvar resultados
    """
    model.to(DEVICE)
    history = {"train_loss": [], "val_loss": []}
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for imgs, labels in train_loader:
            imgs_processed = processor(images=imgs, return_tensors="pt")
            pixel_values = imgs_processed['pixel_values'].to(DEVICE)
            labels = labels.to(DEVICE)
                    
            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        # Validação
        model.eval()
        val_loss = 0.0
        preds, probs, gts = [], [], []
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs_processed = processor(images=imgs, return_tensors="pt")
                pixel_values = imgs_processed['pixel_values'].to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(pixel_values=pixel_values).logits
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                pred = torch.argmax(outputs, dim=1)
                preds.extend(pred.cpu().numpy())
                probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                gts.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        history["val_loss"].append(val_loss)

        acc = accuracy_score(gts, preds)
        prec = precision_score(gts, preds, zero_division=0)
        rec = recall_score(gts, preds, zero_division=0)
        f1 = f1_score(gts, preds, zero_division=0)
        auc = roc_auc_score(gts, probs)

        print(f"📘 Época {epoch+1}/{epochs} | Loss treino: {train_loss:.4f} | Loss val: {val_loss:.4f}")
        print(f"➡️  ACC={acc*100:.2f}% | PREC={prec*100:.2f}% | REC={rec*100:.2f}% | F1={f1*100:.2f}% | AUC={auc*100:.2f}%")

    np.save(f"{save_dir}/probs_{model_name}_fold{fold_idx}.npy", np.array(probs))
    np.save(f"{save_dir}/gts_fold{fold_idx}.npy", np.array(gts))

    return model, (acc, prec, rec, f1, auc)


def kfold_training(data_root="data", model_name="vit", k_fold=5, epochs=10, lr=1e-4, batch_size=32, save_dir="results"):
    """
    Executa o treinamento com validação cruzada K-fold para um modelo específico
    e salva os resultados em um arquivo TXT.
    """
    print(f"\n🚀 Iniciando K-Fold Training ({k_fold} folds) — Modelo: {model_name.upper()}\n")
    dataset = load_dataset(data_root, model_name)
    folds = create_kfold_loaders(dataset, k_fold=k_fold, batch_size=batch_size)

    os.makedirs(save_dir, exist_ok=True)
    txt_file_path = os.path.join(save_dir, f"results_{model_name}.txt")

    all_metrics = []

    with open(txt_file_path, "w") as f:
        f.write(f"📊 Resultados K-Fold ({k_fold} folds) — Modelo: {model_name.upper()}\n\n")

        for fold_idx, (train_loader, val_loader) in enumerate(folds, start=1):
            f.write(f"==============================\n")
            f.write(f"🔹 FOLD {fold_idx}/{k_fold}\n")
            f.write(f"==============================\n")

            processor, model = load_transformer_model(model_name, num_classes=2)
            model.to(DEVICE)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

            trained_model, metrics = train_one_fold(
                model, processor, train_loader, val_loader, optimizer, criterion,
                model_name, fold_idx, epochs, save_dir
            )
            all_metrics.append(metrics)

            acc, prec, rec, f1, auc = metrics
            f.write(f"Accuracy:     {acc*100:.2f}%\n")
            f.write(f"Precision:    {prec*100:.2f}%\n")
            f.write(f"Recall:       {rec*100:.2f}%\n")
            f.write(f"F1-Score:     {f1*100:.2f}%\n")
            f.write(f"AUC:          {auc*100:.2f}%\n\n")

        all_metrics = np.array(all_metrics)
        mean_metrics = all_metrics.mean(axis=0)
        std_metrics = all_metrics.std(axis=0)

        f.write(f"✅ MÉDIA FINAL ({model_name.upper()} — {k_fold} folds):\n")
        f.write(f"Accuracy:     {mean_metrics[0]*100:.2f}% ± {std_metrics[0]*100:.2f}%\n")
        f.write(f"Precision:    {mean_metrics[1]*100:.2f}% ± {std_metrics[1]*100:.2f}%\n")
        f.write(f"Recall:       {mean_metrics[2]*100:.2f}% ± {std_metrics[2]*100:.2f}%\n")
        f.write(f"F1-Score:     {mean_metrics[3]*100:.2f}% ± {std_metrics[3]*100:.2f}%\n")
        f.write(f"AUC:          {mean_metrics[4]*100:.2f}% ± {std_metrics[4]*100:.2f}%\n")

    print(f"\n✅ Resultados salvos em {txt_file_path}")
    return all_metrics


def full_experiment(data_root="data", k_fold=5, epochs=10, lr=1e-4, batch_size=32):
    """
    Executa o experimento completo com todos os modelos Transformers.
    (VIT, DEIT, BEIT, SWINV2) e salva resultados em TXT individual.
    """
    models = ["vit", "deit", "beit", "swinv2"]
    all_results = {}

    for model_name in models:
        print("\n" + "=" * 70)
        print(f"🚀 Iniciando treinamento completo para {model_name.upper()}")
        print("=" * 70)
        results = kfold_training(
            data_root=data_root,
            model_name=model_name,
            k_fold=k_fold,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            save_dir="results"
        )
        all_results[model_name] = results

    print("\n🎯 Treinamento completo finalizado para todos os modelos!")
    return all_results


if __name__ == "__main__":
    full_experiment(
        data_root="data",
        k_fold=5,
        epochs=15,
        lr=1e-4,
        batch_size=16
    )

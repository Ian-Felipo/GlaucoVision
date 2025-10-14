import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

def ensemble_predictions(prob_files_dict, gts_file_dict, save_dir="results"):
    """
    Calcula o Ensemble por média das probabilidades entre múltiplos modelos
    e salva os resultados em um arquivo TXT.

    Parâmetros:
        prob_files_dict (dict): {'vit': ['vit_fold1.npy', 'vit_fold2.npy', ...], ...}
        gts_file_dict (dict): {'vit': ['gts_fold1.npy', ...]} (iguais entre modelos)
        save_dir (str): pasta para salvar o arquivo TXT

    Retorna:
        results (np.array): métricas por fold [acc, prec, rec, f1, auc]
    """
    os.makedirs(save_dir, exist_ok=True)
    txt_file_path = os.path.join(save_dir, "ensemble_results.txt")

    num_folds = len(next(iter(prob_files_dict.values())))
    results = []

    print(f"\n🚀 Iniciando Ensemble com {len(prob_files_dict)} modelos e {num_folds} folds...\n")

    with open(txt_file_path, "w") as f:
        f.write(f"📊 Resultados Ensemble ({len(prob_files_dict)} modelos, {num_folds} folds)\n\n")

        for fold in range(num_folds):
            f.write(f"==============================\n")
            f.write(f"🔹 Fold {fold + 1}/{num_folds}\n")
            f.write(f"==============================\n")

            probs_models = []
            for model_name, files in prob_files_dict.items():
                probs = np.load(files[fold])
                probs_models.append(probs)
                f.write(f"{model_name.upper()} carregado (shape: {probs.shape})\n")
                print(f"  → {model_name.upper()} carregado para o fold {fold + 1} (shape: {probs.shape})")

            probs_mean = np.mean(probs_models, axis=0)
            preds_ensemble = (probs_mean >= 0.5).astype(int)
            gts = np.load(gts_file_dict[list(gts_file_dict.keys())[0]][fold])

            acc = accuracy_score(gts, preds_ensemble)
            prec = precision_score(gts, preds_ensemble)
            rec = recall_score(gts, preds_ensemble)
            f1 = f1_score(gts, preds_ensemble)
            auc = roc_auc_score(gts, probs_mean)

            f.write(f"Accuracy:     {acc*100:.2f}%\n")
            f.write(f"Precision:    {prec*100:.2f}%\n")
            f.write(f"Recall:       {rec*100:.2f}%\n")
            f.write(f"F1-Score:     {f1*100:.2f}%\n")
            f.write(f"AUC:          {auc*100:.2f}%\n\n")

            print(f"  ACC={acc*100:.2f}% | PREC={prec*100:.2f}% | REC={rec*100:.2f}% | F1={f1*100:.2f}% | AUC={auc*100:.2f}%")
            results.append([acc, prec, rec, f1, auc])

        results = np.array(results)
        mean_metrics = results.mean(axis=0)
        std_metrics = results.std(axis=0)

        f.write("📊 MÉTRICAS MÉDIAS DO ENSEMBLE:\n")
        f.write(f"Accuracy:     {mean_metrics[0]*100:.2f}% ± {std_metrics[0]*100:.2f}%\n")
        f.write(f"Precision:    {mean_metrics[1]*100:.2f}% ± {std_metrics[1]*100:.2f}%\n")
        f.write(f"Recall:       {mean_metrics[2]*100:.2f}% ± {std_metrics[2]*100:.2f}%\n")
        f.write(f"F1-Score:     {mean_metrics[3]*100:.2f}% ± {std_metrics[3]*100:.2f}%\n")
        f.write(f"AUC:          {mean_metrics[4]*100:.2f}% ± {std_metrics[4]*100:.2f}%\n")

    print(f"\n✅ Resultados do ensemble salvos em {txt_file_path}")
    return results


if __name__ == "__main__":
    prob_files = {
        "vit": [f"results/probs_vit_fold{i}.npy" for i in range(1, 6)],
        "swinv2": [f"results/probs_swinv2_fold{i}.npy" for i in range(1, 6)],
        "deit": [f"results/probs_deit_fold{i}.npy" for i in range(1, 6)],
        "beit": [f"results/probs_beit_fold{i}.npy" for i in range(1, 6)],
    }

    gts_files = {
        "vit": [f"results/gts_fold{i}.npy" for i in range(1, 6)]
    }

    ensemble_predictions(prob_files, gts_files)

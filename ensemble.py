import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



def ensemble_predictions(prob_files_dict, gts_file_dict):
    """
    Calcula o Ensemble por média das probabilidades entre múltiplos modelos.

    Parâmetros:
        prob_files_dict (dict): {'vit': ['vit_fold1.npy', 'vit_fold2.npy', ...], ...}
        gts_file_dict (dict): {'vit': ['gts_fold1.npy', ...]} (iguais entre modelos)

    Retorna:
        results (list): lista de métricas [acc, prec, rec, f1, auc] por fold
    """
    num_folds = len(next(iter(prob_files_dict.values())))
    results = []

    print(f"\n🚀 Iniciando Ensemble com {len(prob_files_dict)} modelos e {num_folds} folds...\n")

    for fold in range(num_folds):
        print(f"🔹 Avaliando Fold {fold + 1}/{num_folds}")

        probs_models = []
        for model_name, files in prob_files_dict.items():
            probs = np.load(files[fold])
            probs_models.append(probs)
            print(f"  → {model_name.upper()} carregado para o fold {fold + 1} (shape: {probs.shape})")

        probs_mean = np.mean(probs_models, axis=0)
        preds_ensemble = (probs_mean >= 0.5).astype(int)

        gts = np.load(gts_file_dict[list(gts_file_dict.keys())[0]][fold])

        acc = accuracy_score(gts, preds_ensemble)
        prec = precision_score(gts, preds_ensemble)
        rec = recall_score(gts, preds_ensemble)
        f1 = f1_score(gts, preds_ensemble)
        auc = roc_auc_score(gts, probs_mean)

        print(f"  ACC={acc*100:.2f}% | PREC={prec*100:.2f}% | REC={rec*100:.2f}% | F1={f1*100:.2f}% | AUC={auc*100:.2f}%")
        results.append([acc, prec, rec, f1, auc])

    results = np.array(results)
    mean_metrics = results.mean(axis=0)
    std_metrics = results.std(axis=0)

    print("\n📊 MÉTRICAS MÉDIAS DO ENSEMBLE:")
    print(f"  Accuracy:     {mean_metrics[0]*100:.2f}% ± {std_metrics[0]*100:.2f}%")
    print(f"  Precision:    {mean_metrics[1]*100:.2f}% ± {std_metrics[1]*100:.2f}%")
    print(f"  Recall:       {mean_metrics[2]*100:.2f}% ± {std_metrics[2]*100:.2f}%")
    print(f"  F1-Score:     {mean_metrics[3]*100:.2f}% ± {std_metrics[3]*100:.2f}%")
    print(f"  AUC:          {mean_metrics[4]*100:.2f}% ± {std_metrics[4]*100:.2f}%")

    return results


if __name__ == "__main__":
    """
    Exemplo de uso:
    Suponha que você tenha salvo as probabilidades e rótulos de cada modelo
    no formato 'probs_<modelo>_foldX.npy' e 'gts_foldX.npy'.

    Estrutura esperada:
        results/
            ├── probs_vit_fold1.npy
            ├── probs_swinv2_fold1.npy
            ├── probs_deit_fold1.npy
            ├── probs_beit_fold1.npy
            ├── gts_fold1.npy
            └── ...
    """

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

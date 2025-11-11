"""
data_preparation.py
-------------------
Módulo responsável pelo carregamento e preparação dos dados
para o diagnóstico de glaucoma a partir de retinografias.
Este script replica a metodologia descrita no artigo
"Diagnóstico de Glaucoma em Retinografias de Oftalmoscópio Portátil
Utilizando Ensemble Baseado em Transformers" (Costa et al., 2024).

Etapas:
1. Carregamento das imagens com torchvision.datasets.ImageFolder
2. Aplicação de transformações (Resize, Augmentation, Normalize)
3. Implementação manual de validação cruzada K-fold
4. Criação dos DataLoaders para treino e validação
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_transforms(model_name: str):
    """
    Retorna transformações SEM ToTensor() e SEM Normalize()
    O processor fará isso depois.
    """
    input_size = {
        "swinv2": 256,
        "beit": 224,
        "deit": 224,
        "vit": 224
    }.get(model_name.lower(), 224)
    
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        # NÃO adicionar ToTensor() aqui
        # O processor fará a conversão e normalização
    ])
    
    return transform


def load_dataset(data_root: str, model_name: str):
    """
    Carrega o dataset de imagens e aplica as transformações definidas.
    
    Parâmetros:
        data_root (str): caminho da pasta com subpastas /normal /glaucoma
        model_name (str): nome do modelo (para definir transformações)
    
    Retorna:
        dataset (torchvision.datasets.ImageFolder)
    """
    transform = get_transforms(model_name)
    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    return dataset


def create_kfold_loaders(dataset, k_fold=5, batch_size=32, seed=42, collate_fn=None):
    """
    Implementa manualmente a validação cruzada K-fold,
    retornando DataLoaders de treino e validação para cada fold.
    
    Parâmetros:
        dataset: Dataset do torchvision
        k_fold (int): número de folds (padrão = 5)
        batch_size (int): tamanho do batch
        seed (int): semente aleatória para reprodutibilidade
        collate_fn (callable): função customizada para processar batches (opcional)
    
    Retorna:
        folds (list): lista de tuplas (train_loader, val_loader)
    """
    np.random.seed(seed)
    size = len(dataset)
    indices = np.arange(size)
    np.random.shuffle(indices)
    
    split_size = size // k_fold
    folds = []
    
    for fold in range(k_fold):
        val_start = fold * split_size
        val_end = val_start + split_size
        
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate((indices[:val_start], indices[val_end:]))
        
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        # ✅ Adiciona collate_fn aos DataLoaders se fornecido
        train_loader = DataLoader(
            train_subset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_subset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=collate_fn
        )
        
        folds.append((train_loader, val_loader))
        print(f"✅ Fold {fold+1}/{k_fold} criado — treino: {len(train_subset)} | validação: {len(val_subset)}")
    
    return folds


if __name__ == "__main__":
    # Exemplo de execução isolada
    dataset = load_dataset("data", model_name="vit")
    folds = create_kfold_loaders(dataset, k_fold=5, batch_size=32)
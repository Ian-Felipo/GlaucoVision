import torch
import torch.nn as nn
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification
)

MODEL_CONFIGS = {
    "vit": {
        "name": "google/vit-base-patch16-224",
        "input_size": 224
    },
    "swinv2": {
        "name": "microsoft/swinv2-base-patch4-window16-256",
        "input_size": 256
    },
    "deit": {
        "name": "facebook/deit-base-distilled-patch16-224",
        "input_size": 224
    },
    "beit": {
        "name": "microsoft/beit-base-patch16-224-pt22k-ft22k",
        "input_size": 224
    }
}


def load_transformer_model(model_name: str, num_classes: int = 2):
    """
    Carrega o modelo Transformer pré-treinado e adapta para classificação binária.

    Parâmetros:
        model_name (str): nome do modelo ('vit', 'swinv2', 'deit', 'beit')
        num_classes (int): número de classes (default = 2)

    Retorna:
        processor (AutoImageProcessor): pré-processador de imagem correspondente
        model (nn.Module): modelo ajustado para classificação binária
    """
    model_name = model_name.lower()
    assert model_name in MODEL_CONFIGS, f"Modelo '{model_name}' não suportado."

    cfg = MODEL_CONFIGS[model_name]
    print(f"🔹 Carregando modelo: {cfg['name']}")

    processor = AutoImageProcessor.from_pretrained(cfg["name"])
    model = AutoModelForImageClassification.from_pretrained(
        cfg["name"],
        num_labels=num_classes,
        ignore_mismatched_sizes=True  
    )

    if hasattr(model, "classifier"):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, "head"):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    elif hasattr(model, "heads"):
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    return processor, model


def get_all_models(num_classes: int = 2):
    """
    Retorna todos os modelos usados no ensemble, já configurados.

    Retorna:
        models (dict): {'vit': model, 'swinv2': model, ...}
        processors (dict): {'vit': processor, ...}
    """
    models = {}
    processors = {}

    for name in MODEL_CONFIGS.keys():
        processor, model = load_transformer_model(name, num_classes=num_classes)
        models[name] = model
        processors[name] = processor

    return processors, models


if __name__ == "__main__":
    processors, models = get_all_models()
    for name, model in models.items():
        print(f"{name.upper()} carregado com sucesso. Parâmetros: {sum(p.numel() for p in model.parameters()):,}")

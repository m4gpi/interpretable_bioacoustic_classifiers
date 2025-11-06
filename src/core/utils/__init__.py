import torch

from typing import Callable, List

def weight_initialization(
    in_features: int,
    dataloader: torch.utils.data.DataLoader,
    species_names: List[str],
    device: str,
    loss: Callable,
    num_initialisations: int = 100,
) -> torch.nn.ModuleDict:
    """
    Create a set of num_initialisations linear classifiers for each species and evaluate
    performance on the training set. Pick the best linear classifier for each species to use as an initialisation point.
    """
    classifiers = torch.nn.ModuleDict({})
    # J initialisations for each species
    for species_name in species_names:
        classifiers[species_name] = torch.nn.ModuleList([
            torch.nn.Linear(in_features=in_features, out_features=1, bias=True)
            for i in range(num_initialisations)
        ])
    classifiers = classifiers.to(device)
    # evaluate k layers on entire training set, select the best performing model on the training set
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x, y, _ = batch
            x = x.to(device)
            y = y.to(device)
            for j in range(num_initialisations):
                # evaluate every layer with this example
                logits = torch.cat([
                    torch.max(layers[j](x), dim=1).values
                    for layers in classifiers.values()
                ], dim=-1)
                losses.append(loss(y.float(), logits).t())
    # sum the loss across all examples
    indices = torch.stack(losses, dim=1).sum(dim=1).argmin(dim=-1)
    # for each species select the layer within its subset of layers that yields the lowest loss
    for idx, (species_name, layers) in zip(indices % num_initialisations, classifiers.items()):
        classifiers[species_name] = layers[idx]
    return classifiers


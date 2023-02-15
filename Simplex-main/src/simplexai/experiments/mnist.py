import argparse
import math
import os
import pickle as pkl
import time
from pathlib import Path

import captum.attr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_influence_functions as ptif
import seaborn as sns
import sklearn
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset

from simplexai.explainers.nearest_neighbours import NearNeighLatent
from simplexai.explainers.representer import Representer
from simplexai.explainers.simplex import Simplex
from simplexai.explainers.contex import Contex
from simplexai.models.image_recognition import MnistClassifier
from simplexai.utils.schedulers import ExponentialScheduler


# Load data
class MNISTSubset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> tuple:
        if torch.is_tensor(i):
            i = i.tolist()
        return self.X[i], self.y[i]


def load_mnist(
    batch_size: int, train: bool, subset_size=None, shuffle: bool = True
) -> DataLoader:
    dataset = torchvision.datasets.MNIST(
        "./data/",
        train=train,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )
    if subset_size:
        dataset = torch.utils.data.Subset(
            dataset, torch.randperm(len(dataset))[:subset_size]
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_emnist(batch_size: int, train: bool) -> DataLoader:
    return DataLoader(
        torchvision.datasets.EMNIST(
            "./data/",
            train=train,
            download=True,
            split="letters",
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    
def train_model(
    save_path: Path,
    device: torch.device,
    #n_epoch: int = 10,
    #batch_size_train: int = 64,
    #batch_size_test: int = 1000,
    n_epoch: int = 4,
    batch_size_train: int = 200,
    batch_size_test: int = 1000,
    random_seed: int = 42,
    learning_rate: float = 0.01,
    momentum: float = 0.5,
    log_interval: int = 100,
    model_reg_factor: float = 0.01,
    cv: int = 0,
) -> MnistClassifier:
    torch.random.manual_seed(random_seed + cv)
    torch.backends.cudnn.enabled = False

    # Prepare data
    train_loader = load_mnist(batch_size_train, train=True)
    test_loader = load_mnist(batch_size_test, train=False)

    # Create the model
    classifier = MnistClassifier()
    classifier.to(device)
    optimizer = optim.SGD(
        classifier.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=model_reg_factor,
    )

    # Train the model
    train_losses = []
    train_counter = []
    test_losses = []

    def train(epoch):
        classifier.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = classifier(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                    f" ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
                )
                torch.save(
                    classifier.state_dict(),
                    os.path.join(save_path, f"model_cv{cv}.pth"),
                )
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(save_path, f"optimizer_cv{cv}.pth"),
                )

    def test():
        classifier.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)
                output = classifier(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print(
            f"\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}"
            f"({100. * correct / len(test_loader.dataset):.0f}%)\n"
        )

    test()
    for epoch in range(1, n_epoch + 1):
        train(epoch)
        test()
    torch.save(classifier.state_dict(), os.path.join(save_path, f"model_cv{cv}.pth"))
    torch.save(optimizer.state_dict(), os.path.join(save_path, f"optimizer_cv{cv}.pth"))
    return classifier


def fit_explainers(
    device: torch.device,
    explainers_name: list,
    save_path: Path,
    corpus_size: int = 1000,
    test_size: int = 100,
    n_epoch: int = 10000,
    random_seed: int = 42,
    n_keep: int = 5,
    reg_factor_init: float = 0.1,
    reg_factor_final: float = 100,
    cv: int = 0,
    train_only: bool = False,
) -> list:
    torch.random.manual_seed(random_seed + cv)
    explainers = []

    # Load model:
    classifier = MnistClassifier()
    classifier.load_state_dict(torch.load(save_path / f"model_cv{cv}.pth"))
    classifier.to(device)
    classifier.eval()

    # Load data:
    corpus_loader = load_mnist(corpus_size, train=True)
    test_loader = load_mnist(test_size, train=train_only)
    corpus_examples = enumerate(corpus_loader)
    test_examples = enumerate(test_loader)
    batch_id_test, (test_data, test_targets) = next(test_examples)
    batch_id_corpus, (corpus_data, corpus_target) = next(corpus_examples)
    corpus_data = corpus_data.to(device).detach()
    test_data = test_data.to(device).detach()
    corpus_latent_reps = classifier.latent_representation(corpus_data).detach()
    corpus_probas = classifier.probabilities(corpus_data).detach()
    corpus_true_classes = torch.zeros(corpus_probas.shape, device=device)
    corpus_true_classes[torch.arange(corpus_size), corpus_target] = 1
    test_latent_reps = classifier.latent_representation(test_data).detach()

    # Fit ContEx:
    reg_factor_scheduler = ExponentialScheduler(
        reg_factor_init, reg_factor_final, n_epoch
    )
    contex = Contex(
        corpus_examples=corpus_data, corpus_latent_reps=corpus_latent_reps
    )

    contex.construct_corpus()

    contex.fit(
        test_examples=test_data,
        test_latent_reps=test_latent_reps,
        n_epoch=n_epoch,
        reg_factor=reg_factor_init,
        n_keep=n_keep,
        reg_factor_scheduler=reg_factor_scheduler,
    )
    explainers.append(contex)

    # Fit SimplEx:
    reg_factor_scheduler = ExponentialScheduler(
        reg_factor_init, reg_factor_final, n_epoch
    )
    simplex = Simplex(
        corpus_examples=corpus_data, corpus_latent_reps=corpus_latent_reps
    )
    simplex.fit(
        test_examples=test_data,
        test_latent_reps=test_latent_reps,
        n_epoch=n_epoch,
        reg_factor=reg_factor_init,
        n_keep=n_keep,
        reg_factor_scheduler=reg_factor_scheduler,
    )
    explainers.append(simplex)

    # Fit nearest neighbors:
    nn_uniform = NearNeighLatent(
        corpus_examples=corpus_data, corpus_latent_reps=corpus_latent_reps
    )
    nn_uniform.fit(
        test_examples=test_data, test_latent_reps=test_latent_reps, n_keep=n_keep
    )
    explainers.append(nn_uniform)
    nn_dist = NearNeighLatent(
        corpus_examples=corpus_data,
        corpus_latent_reps=corpus_latent_reps,
        weights_type="distance",
    )
    nn_dist.fit(
        test_examples=test_data, test_latent_reps=test_latent_reps, n_keep=n_keep
    )
    explainers.append(nn_dist)

    # Save explainers and data:
    for explainer, explainer_name in zip(explainers, explainers_name):
        explainer_path = save_path / f"{explainer_name}_cv{cv}_n{n_keep}.pkl"
        with open(explainer_path, "wb") as f:
            print(f"Saving {explainer_name} decomposition in {explainer_path}.")
            pkl.dump(explainer, f)
    corpus_data_path = save_path / f"corpus_data_cv{cv}.pkl"
    with open(corpus_data_path, "wb") as f:
        print(f"Saving corpus data in {corpus_data_path}.")
        pkl.dump([corpus_latent_reps, corpus_probas, corpus_true_classes], f)
    test_data_path = save_path / f"test_data_cv{cv}.pkl"
    with open(test_data_path, "wb") as f:
        print(f"Saving test data in {test_data_path}.")
        pkl.dump([test_latent_reps, test_targets], f)
    return explainers


def fit_representer(
    model_reg_factor: float, load_path: Path, cv: int = 0
) -> Representer:
    # Fit the representer explainer (this is only makes sense by using the whole corpus)
    corpus_data_path = os.path.join(load_path, f"corpus_data_cv{cv}.pkl")
    with open(corpus_data_path, "rb") as f:
        corpus_latent_reps, corpus_probas, corpus_true_classes = pkl.load(f)
    test_data_path = os.path.join(load_path, f"test_data_cv{cv}.pkl")
    with open(test_data_path, "rb") as f:
        test_latent_reps, test_targets = pkl.load(f)
    representer = Representer(
        corpus_latent_reps=corpus_latent_reps,
        corpus_probas=corpus_probas,
        corpus_true_classes=corpus_true_classes,
        reg_factor=model_reg_factor,
    )
    representer.fit(test_latent_reps=test_latent_reps)
    explainer_path = os.path.join(load_path, f"representer_cv{cv}.pkl")
    with open(explainer_path, "wb") as f:
        print(f"Saving representer decomposition in {explainer_path}.")
        pkl.dump(representer, f)
    return representer



def approximation_quality(
    n_keep_list: list,
    cv: int = 0,
    random_seed: int = 42,
    model_reg_factor=0.1,
    save_path: str = "results/mnist/quality/",
    train_only=False,
) -> None:
    print(
        100 * "-"
        + "\n"
        + "Welcome in the approximation quality experiment for MNIST. \n"
        f"Settings: random_seed = {random_seed} ; cv = {cv}.\n" + 100 * "-"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #explainers_name = ["simplex", "nn_uniform", "nn_dist", "representer"]
    explainers_name = ["contex", "simplex", "nn_uniform", "nn_dist"]
    print(explainers_name)
    current_path = Path.cwd()
    save_path = current_path / save_path
    # Create saving directory if inexistent
    if not save_path.exists():
        print(f"Creating the saving directory {save_path}")
        os.makedirs(save_path)

    # Training a model, save it
    print(100 * "-" + "\n" + "Now fitting the model. \n" + 100 * "-")
    train_model(
        device=device,
        random_seed=random_seed,
        cv=cv,
        save_path=save_path,
        model_reg_factor=model_reg_factor,
    )

    # Load the model
    classifier = MnistClassifier()
    classifier.load_state_dict(torch.load(save_path / f"model_cv{cv}.pth"))
    classifier.to(device)
    classifier.eval()

    # Fit the explainers
    print(100 * "-" + "\n" + "Now fitting the explainers. \n" + 100 * "-")
    for i, n_keep in enumerate(n_keep_list):
        print(30 * "-" + f"n_keep = {n_keep}" + 30 * "-")
        explainers = fit_explainers(
            device=device,
            random_seed=random_seed,
            cv=cv,
            test_size=100,
            corpus_size=1000,
            n_keep=n_keep,
            save_path=save_path,
            explainers_name=explainers_name,
            train_only=train_only,
        )
        # Print the partial results
        print(100 * "-" + "\n" + "Results. \n" + 100 * "-")
        #for explainer, explainer_name in zip(explainers, explainers_name[:-1]):
        for explainer, explainer_name in zip(explainers, explainers_name):
            latent_rep_approx = explainer.latent_approx()
            latent_rep_true = explainer.test_latent_reps
            output_approx = classifier.latent_to_presoftmax(latent_rep_approx).detach()
            output_true = classifier.latent_to_presoftmax(latent_rep_true).detach()
            latent_r2_score = sklearn.metrics.r2_score(
                latent_rep_true.cpu().numpy(), latent_rep_approx.cpu().numpy()
            )
            output_r2_score = sklearn.metrics.r2_score(
                output_true.cpu().numpy(), output_approx.cpu().numpy()
            )
            print(
                f"{explainer_name} latent r2: {latent_r2_score:.2g} ; output r2 = {output_r2_score:.2g}."
            )

    
    # Fit the representer explainer (this is only makes sense by using the whole corpus)
    representer = fit_representer(model_reg_factor, save_path, cv)
    latent_rep_true = representer.test_latent_reps
    output_true = classifier.latent_to_presoftmax(latent_rep_true).detach()
    output_approx = representer.output_approx()
    output_r2_score = sklearn.metrics.r2_score(
        output_true.cpu().numpy(), output_approx.cpu().numpy()
    )
    print(f"representer output r2 = {output_r2_score:.2g}.")



def main(experiment: str, cv: int) -> None:
    if experiment == "approximation_quality":
        approximation_quality(cv=cv, n_keep_list=[3, 5, 10, 20, 50])
    else:
        raise ValueError(
            "The name of the experiment is not valid. "
            "Valid names are: "
            "approximation_quality , outlier_detection , jacobian_corruption, influence, timing."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-experiment",
        type=str,
        default="approximation_quality",
        help="Experiment to perform",
    )
    parser.add_argument("-cv", type=int, default=0, help="Cross validation parameter")
    args = parser.parse_args()
    main(args.experiment, args.cv)

import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

from torch_geometric.loader import DataLoader
from collections import Counter
import torch
import joblib

from models.mlp_classifier import MLPClassifier
from train.train_eval_mlp import evaluate
from utils.io_tools import load_yaml, seed_everything
from utils.visual_utils import plot_confusion_matrix, plot_per_class_accuracy, plot_confused_classes_only
from utils.compute_metrics import compute_classification_metrics, compute_per_class_accuracy


YELLOW = "\033[93m"
RESET = "\033[0m"
seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data_and_model():

    root_dir = "dataset"
    batch_size = 256
    hidden_dim = 256

    # Load dataset config
    config = load_yaml(f"config/data_config.yaml")
    global_perf_list = list(config["Performance"].keys())
    circuit_list = list(config["Classes"].keys())

    # Load dataset
    data = torch.load(f"{root_dir}/performance_class_data.pt", weights_only=True)
    test_dataset = data["test"]
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    test_labels = [y for _, y in test_dataset]
    print(f"{YELLOW}[DEBUG] Test class distribution:", dict(sorted(Counter(test_labels).items())), f"{RESET}")

    # Load scaler
    scaler_path = f"./{root_dir}/performance_scaler_mlp.pkl"
    scalers = joblib.load(scaler_path)

    # Load trained model
    input_dim = len(global_perf_list)
    output_dim = len(circuit_list)

    model = MLPClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    model.load_state_dict(torch.load("./checkpoints/best_mlp_model.pt", map_location="cpu", weights_only=True))
    model.eval()

    return model, test_loader, scalers, circuit_list, global_perf_list


if __name__ == "__main__":
    model, test_loader, scalers, circuit_list, global_perf_list = load_data_and_model()
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, device, return_preds=True)
    print(f"[RESULT] Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    per_class_table, summary_table = compute_classification_metrics(test_labels, test_preds, circuit_list)
    per_class_table.to_csv("./results/mlp_per_class_metrics.csv")
    summary_table.to_csv("./results/mlp_summary_metrics.csv", index=False)

    print(per_class_table.to_markdown())
    print(summary_table.to_markdown())

    # Plot confusion matrix
    plot_confusion_matrix(
        y_true=test_labels,
        y_pred=test_preds,
        class_names=circuit_list,
        normalize=True,
        save_path='./plots/mlp/ConfusionMatrix.pdf'
    )

    plot_confused_classes_only(test_labels, test_preds, circuit_list, 
                            title="Confusion Among Misclassified Topologies", 
                            save_path="./plots/mlp/MisclassifiedMatrix.pdf")


    class_accuracies = compute_per_class_accuracy(test_labels, test_preds, num_classes=len(circuit_list))
    plot_per_class_accuracy(class_accuracies, class_names=circuit_list, save_path='./plots/mlp/PerClassAccuracy.pdf')
# from model import MyAwesomeModel
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# from dtu_mlops_code import MyAwesomeModel
import sys

# sys.path.append("../models/")
# import dtu_mlops_code
from dtu_mlops_code.models.model import MyAwesomeModel


def visualize_features(model_path, data, output_file):
    # Load the pre-trained MyAwesomeNetwork
    # Load the model
    model = MyAwesomeModel()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()

    # Extract the features just before the final classification layer
    features = model.extract_features(data)

    # Use TSNE to reduce the dimensionality of the features to 2D
    tsne = TSNE(n_components=2)
    features_2d = tsne.fit_transform(features)

    # Visualize the 2D features
    plt.scatter(features_2d[:, 0], features_2d[:, 1])
    plt.title("Features Visualization")

    # Save the visualization to a file in the reports/figures/ folder
    plt.savefig(os.path.join("./reports/figures/", output_file))


if __name__ == "__main__":
    # Usage
    import os

    print("\n" + os.getcwd() + "\n")
    data = torch.load("./data/processed/test_images.pt")
    visualize_features("./models/trained_model.pt", data, "visualization_test.png")

# python dtu_mlops_code/visualizations/visualize.py

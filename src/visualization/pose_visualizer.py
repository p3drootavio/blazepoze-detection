import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class PoseVisualizer():
    """
    PoseVisualizer

    A class for visualizing pose data and their distributions using various dimensionality reduction
    and plotting techniques.

    This class provides methods to visualize pose landmarks data through different representations
    including PCA, t-SNE, and temporal analysis. It works with encoded pose data and supports
    various visualization options.

    Parameters
    ----------
    pipeline : object
        A pipeline object that contains the following attributes:
        - encoder: Encoder model for pose data
        - X: Input pose data
        - y_encoder: Encoded labels
        - X_train: Training data
        - y_train: Training labels
        - data_frames_list: List of pose data frames

    Attributes
    ----------
    pipeline : object
        The input pipeline object
    encoder : object
        Encoder model from the pipeline
    X : array-like
        Input pose data
    y_encoder : array-like
        Encoded labels
    X_train : array-like
        Training data
    y_train : array-like
        Training labels
    data_frames_list : list
        List containing pose data frames

    Methods
    -------
    plot_landmarks_distribution(sample_size=100, save_fig=False, save_path=None, grid_on=False)
        Visualize pose data distribution using PCA dimensionality reduction.

    plot_landmarks_across_time(pose=0, joint_index=0, save_fig=False, save_path=None, grid_on=False)
        Visualize landmark trajectories over time for selected poses and joints.

    plot_landmarks_clustered(save_fig=False, save_path=None, grid_on=False)
        Visualize landmark data clustering using t-SNE dimensionality reduction.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.encoder = pipeline.encoder
        self.X = pipeline.X
        self.y_encoder = pipeline.y_encoder
        self.X_train = pipeline.X_train
        self.y_train = pipeline.y_train
        self.data_frames_list = pipeline.data_frames_list


    def plot_landmarks_distribution(self, sample_size=100, save_fig=False, save_path=None, grid_on=False):
        """
        Visualize pose data distribution using PCA dimensionality reduction.

        This method creates a figure with two subplots: a scatter plot and a bar plot
        showing the distribution of pose landmarks after PCA reduction to 2D.

        Parameters
        ----------
        sample_size : int, optional
            Number of samples to visualize, by default 100
        save_fig : bool, optional
            Whether to save the figure to disk, by default False
        save_path : str, optional
            Path where to save the figure if save_fig is True, by default None
        grid_on : bool, optional
            Whether to display grid lines on the plots, by default False

        Returns
        -------
        None
            Displays the plot and optionally saves it to disk
        """
        x_vis = self.X_train[:sample_size]
        y_vis = self.y_train[:sample_size]

        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        dataset = tf.data.Dataset.from_tensor_slices((x_vis, y_vis))

        # Collect the dataset into numpy arrays
        X_values, y_values = [], []
        for x, y in dataset.take(sample_size).as_numpy_iterator():
            X_values.append(x)
            y_values.append(y)

        X_np_values = np.array(X_values)
        y_np_values = np.array(y_values)

        # Flatten each sequence: shape becomes (n_samples, sequence_length * n_features)
        X_flat = X_np_values.reshape(len(X_values), -1)

        # Normalize X
        X_flat = self._normalize_data(X_flat, method='std')

        # Reduce to 2D array
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_flat)

        # Label Legend and Plot
        classes = np.unique(y_np_values)
        colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))

        for i, class_id in enumerate(classes):
            idx = y_np_values == class_id
            label_name = self.encoder.inverse_transform([class_id])[0]
            ax1.scatter(X_2d[idx, 0], X_2d[idx, 1], label=label_name, color=colors[i], alpha=0.6, s=100, marker="*")
            ax2.bar(X_2d[idx, 0], X_2d[idx, 1], label=label_name, color=colors[i], alpha=0.6)

        for ax in [ax1, ax2]:
            if grid_on: ax.grid()
            ax.set_xlabel("Principal Component 1", fontstyle='italic', color='white')
            ax.set_ylabel("Principal Component 2", fontstyle='italic', color='white')

        ax1.set_title("Scatter Graph", fontsize=16)
        ax2.set_title("Bar Graph", fontsize=16)
        fig.suptitle("PCA of Pose Sequences", fontsize=20)
        ax1.legend(loc='lower left')
        ax2.legend(loc='lower left')
        plt.tight_layout()
        plt.show()
        plt.close()

        if save_fig and save_path:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(f'{save_path}/pose_sample_plot.png', dpi=300)


    def plot_landmarks_across_time(self, pose=0, joint_index=0, save_fig=False, save_path=None, grid_on=False):
        """
        Visualize landmark trajectories over time for selected poses and joints.

        Creates a line plot showing the X, Y, Z coordinates of a specific joint
        across all frames for a given pose sequence.

        Parameters
        ----------
        pose : int, optional
            Index of the pose sequence to visualize, by default 0
        joint_index : int, optional
            Index of the joint to track (0-32), by default 0
        save_fig : bool, optional
            Whether to save the figure to disk, by default False
        save_path : str, optional
            Path where to save the figure if save_fig is True, by default None
        grid_on : bool, optional
            Whether to display grid lines on the plot, by default False

        Returns
        -------
        None
            Displays the plot and optionally saves it to disk
        """
        if pose >= len(self.data_frames_list):
            print("Error: Invalid pose number.")
            return

        X_vals, Y_vals, Z_vals = [], [], []
        for frame in self.data_frames_list[pose]:  # 50 iterations for the selected data frame of a shape
            reshaped = frame.reshape((33, 3))  # Access [:, 0], [:, 1], [:, 2]
            X_vals.append(reshaped[joint_index:, 0])
            Y_vals.append(reshaped[joint_index:, 1])
            Z_vals.append(reshaped[joint_index:, 2])

        # Plot the mean of X, Y, Z across time (frames)
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 6))
        plt.plot(X_vals, label='X', color='r')
        plt.plot(Y_vals, label='Y', color='g')
        plt.plot(Z_vals, label='Z', color='b')
        plt.title("Average X, Y, Z Components Over Time")
        plt.xlabel("Frame")
        plt.ylabel("Mean Coordinate Value")
        plt.legend()
        plt.grid(grid_on)
        plt.tight_layout()
        plt.show()
        plt.close()

        if save_fig and save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f'{save_path}/pose_sample_plot.png', dpi=300)


    def plot_landmarks_clustered(self, save_fig=False, save_path=None, grid_on=False):
        """
        Visualize landmark data clustering using t-SNE dimensionality reduction.

        Creates a scatter plot of pose landmarks in 2D space after applying t-SNE
        dimensionality reduction, with different colors representing different pose classes.

        Parameters
        ----------
        save_fig : bool, optional
            Whether to save the figure to disk, by default False
        save_path : str, optional
            Path where to save the figure if save_fig is True, by default None
        grid_on : bool, optional
            Whether to display grid lines on the plot, by default False

        Returns
        -------
        None
            Displays the plot and optionally saves it to disk
        """
        X_flat = self.X.reshape(self.X.shape[0], -1)

        # Apply t-SNE on all samples (e.g., 1000 samples of shape (99,))
        tsne = TSNE(n_components=2, random_state=0)
        X_2d = tsne.fit_transform(X_flat)
        y_values = self.y_encoder  # Must be a list/array with same length as self.data_frames_list

        plt.style.use('dark_background')
        plt.figure(figsize=(10, 6))

        # Label Legend and Plot
        classes = np.unique(y_values)
        colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))

        for i, class_id in enumerate(classes):
            idx = np.where(y_values == class_id)
            label_name = self.encoder.inverse_transform([class_id])[0]
            plt.scatter(X_2d[idx, 0], X_2d[idx, 1],
                        label=label_name, color=colors[i], alpha=0.6, s=100, marker="*")

        plt.title("t-SNE Projection of Pose Embeddings")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend()
        plt.grid(grid_on)
        plt.tight_layout()
        plt.show()
        plt.close()

        if save_fig and save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f'{save_path}/pose_sample_plot.png', dpi=300)


    def _normalize_data(self, data, method='min_max'):
        """
        Normalize input data using specified method.

        Args:
            data: Input data to normalize.
            method (str, optional): Normalization method - 'min_max' or 'std'. Defaults to 'min_max'.

        Returns:
            tf.Tensor: Normalized data.

        Notes:
            - min_max scales to [0,1] range
            - std performs standardization (zero mean, unit variance)
        """
        if method == 'min_max':
            min_val = np.min(data)
            max_val = np.max(data)
            return (data - min_val) / (max_val - min_val)

        if method == 'std':
            mean_val = np.mean(data)
            std_val = np.std(data)
            return (data - mean_val) / std_val

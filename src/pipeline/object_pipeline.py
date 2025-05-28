import os
import pickle
import textwrap
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from collections import defaultdict

from pandas.errors import EmptyDataError
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from src.utils import data_augmentation
from src.utils import track_calls
from src.utils import validators


class PoseDatasetPipeline():
    """
    PoseDatasetPipeline

    This class implements a robust pipeline for processing, augmenting, and visualizing pose estimation data. The data is assumed to be stored in sequences of landmarks with a certain dimensionality. The pipeline facilitates loading the data from a directory, splitting it into training, validation, and test datasets, and preparing it for use with TensorFlow models. Additionally, it provides functionality for data augmentation, visualization, and saving/loading the pipeline configuration.

    Attributes:
        data_dir (str): Path to the directory where dataset resides.
        sequence_length (int): Length of each sequence of frames to be processed.
        landmarks_dim (int): Dimensionality of the landmark features in input data.
        batch_size (int): Number of samples per batch for training and validation.
        augmentation_config (dict): Configuration for data augmentation, where keys are augmentation types (e.g., noise, scale, shift) and values are arrays with a boolean flag and a float representing the probability of applying the augmentation.
        augmentation_counts (defaultdict): Tracks the count of augmentations applied during the pipeline.
        X (np.ndarray): Loaded input data.
        y_encoder (np.ndarray): Encoded labels corresponding to the input data.
        encoder (LabelEncoder): Encoder to transform labels between string and integer representation.

    Methods:
        __init__(data_dir, sequence_length, landmarks_dim, batch_size, augmentation_config):
            Initializes the pipeline with given parameters. Validates the format of the augmentation configuration.

        __str__():
            Returns a string representation of the pipeline that includes information about data directory, sequence dimensions, batch size, augmentation, and data loading status.

        load_data():
            Loads pose landmark data from the specified directory (`data_dir`) and prepares it for training. Performs basic validation on sequence shapes and encodes class labels.

        split_data():
            Splits the loaded data into training, validation, and test sets. Retains stratification of class distributions.

        plot_sample(sample_size=50000, augment=False, save_fig=False, grid_on=False):
            Visualizes a subsample of the pose data using PCA. Includes scatter and bar plots for the transformed data, with an option for augmentation and saving the figure.

        get_tf_dataset(split="train", augment=False):
            Returns a TensorFlow dataset corresponding to a specified split (train, val, or test). Provides an option to apply augmentation during dataset creation.

        save_pipeline(save_path, save_config=False):
            Saves the TensorFlow dataset and optional pipeline configuration to storage at the specified path.

        load_pipeline(file_path, load_config=False, file_path_config=''):
            Loads a previously saved TensorFlow dataset and optionally loads saved pipeline configuration from a JSON file.

        _augment_tn_factory():
            Internal method to define and return augmentation transformations for the dataset.
    """

    # Make output stable
    np.random.seed(42)
    tf.random.set_seed(42)


    def __init__(self, data_dir, sequence_length, landmarks_dim, batch_size, augmentation_config):
        """
        Initialize the PoseDatasetPipeline.

        Args:
            data_dir (str): Directory path containing the dataset files.
            sequence_length (int): Length of each sequence of frames.
            landmarks_dim (int): Dimensionality of the landmark features.
            batch_size (int): Number of samples per batch.
            augmentation_config (dict): Configuration for data augmentation with format
                {'augmentation_type': [bool, float]} where bool indicates if enabled
                and float is probability in range [0-1].

        Raises:
            Exception: If augmentation_config format is invalid.
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.landmarks_dim = landmarks_dim
        self.batch_size = batch_size
        self.augmentation_counts = defaultdict(int)
        self.fail = False
        self.errors = []

        if validators.check_instance(augmentation_config):
            self.augmentation_config = augmentation_config
        else:
            raise Exception("""An error occurred when trying to configure augmentation specifications.
            Make sure the attribute augmentation_config follows the following format:
            {'augmentation': [bool, float]} where augmentation is noise, scale, or shift and float is [0-1]
            """)


    def __str__(self):
        """
        Generate a string representation of the pipeline.

        Returns:
            str: Formatted string containing pipeline information including:
                - Basic configuration (data directory, dimensions, batch size)
                - Data loading status and statistics (if data is loaded)
                - Pipeline state indicators (load/split/dataset creation status)
        """

        if self.fail:
            return "An error occurred when trying to generate a string representation of the pipeline."
        relevant_metadata = textwrap.dedent(f"""\
            Pipeline Basic Information:
            Data Directory Path: {self.data_dir}
            Sequence Length: {self.sequence_length}
            Landmarks Dimensions: {self.landmarks_dim}
            Batch Size: {self.batch_size}
            Data Augmentation Configuration: {self.augmentation_config.items()}
        """)

        if self.load_data.has_been_called:
            loaded_data = textwrap.dedent(f"""\
                Pipeline Data Information:
                Size of the loaded data: {self.X.shape}
                Number of gesture classes: {len(np.unique(self.y_encoder))}
                Augmentation Counts: {self.augmentation_counts}
            """)
        else:
            loaded_data = "Data was not loaded yet. No information about data is available!\n\n"

        load_check = "✅" if self.load_data.has_been_called else "❌"
        split_check = "✅" if self.split_data.has_been_called else "❌"
        dataset_check = "✅" if self.get_tf_dataset.has_been_called else "❌"

        description_line = f"PoseDatasetPipeline (Loaded: {load_check} | Split: {split_check} | Dataset Created: {dataset_check})"

        return relevant_metadata + loaded_data + description_line


    @track_calls.trackcalls
    def load_data(self):
        """
        Load pose landmark data from the specified directory.

        Reads CSV files containing landmark data from the data directory,
        processes them into sequences, and encodes class labels.

        Notes:
            - Expects files ending with '_landmarks.csv'
            - Validates sequence shapes against specified dimensions
            - Sets self.X as numpy array of shape (samples, sequence_length, landmarks_dim)
            - Sets self.y_encoder as encoded labels using LabelEncoder

        Raises:
            AssertionError: If loaded data doesn't match expected shape (50, 99)
        """
        X_data, y_labels = [], []
        file_config = ("d__landmarks.csv", "p__landmarks.csv", "ft__landmarks.csv", "ht__landmarks.csv")

        try:
            for folder in os.listdir(self.data_dir):  # Navigate each subfolder
                folder_path = os.path.join(self.data_dir, folder)

                if not os.path.isdir(folder_path): continue  # Skip if it's not a folder


                for file in os.listdir(folder_path):  # Navigate each folder per subfolder
                    file_path = os.path.join(folder_path, file)
                    if (file.endswith(file_config)):
                        pose_df = pd.read_csv(file_path)

                        if pose_df.shape != (self.sequence_length, self.landmarks_dim + 1): continue  # +1 for 'Unnamed: 0'

                        frame_data = pose_df.drop(columns=pose_df.columns[0]).values.astype(np.float32)  # Shape: (50, 99)

                        X_data.append(frame_data)

                        label = file.split("__")[0]
                        y_labels.append(label)

            # Transform data frames to np arrays
            self.X = np.array(X_data, dtype=np.float32)  # Shape: (samples, 50, 99)
            y = np.array(y_labels)

            # Encoder labels
            self.encoder = LabelEncoder()
            self.y_encoder = self.encoder.fit_transform(y)

            assert self.X.shape[1:] == (50, 99), "X does not satisfy the (50, 99) shape"

        except FileNotFoundError:
            print(f"Error: File not found")
            self.fail = True
            self.errors.append("File not found")
        except EmptyDataError:
            print(f"Error: Empty data in file")
            self.fail = True
            self.errors.append("Empty data in file")
        except pd.errors.ParserError as e:
            print(f"Error: Parser error: {e}")
            self.fail = True
            self.errors.append(f"Parser error: {e}")
        except Exception as e:
            print(f"Error: Unknown error occurred: {e}")
            self.fail = True
            self.errors.append(f"Unknown error occurred: {e}")


    @track_calls.trackcalls
    def split_data(self, test_size=0.2, valid_size=0.1):
        """
        Split the loaded data into training, validation, and test sets.

        Performs two-stage split:
            1. 80% training+validation, 20% test
            2. From training+validation: 90% training, 10% validation

        Notes:
            - Maintains class distribution through stratification
            - Sets X_train, X_valid, X_test and corresponding y_ attributes
        """
        if self.fail:
            print(f"Splitting data process was stopped due to {self.errors}. Please check the logs for more information.")
            return

        X_train_full, self.X_test, y_train_full, self.y_test = train_test_split(self.X, self.y_encoder,
                                                                                test_size=test_size, stratify=self.y_encoder)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X_train_full, y_train_full,
                                                                                  test_size=valid_size)


    def plot_sample(self, sample_size=100, augment=False, save_fig=False, save_path=None, grid_on=False):
        """
        Visualize pose data using PCA dimensionality reduction.

        Args:
            sample_size (int, optional): Number of samples to visualize. Defaults to 50000.
            augment (bool, optional): Whether to apply augmentation before visualization. Defaults to False.
            save_fig (bool, optional): Whether to save the plot to file. Defaults to False.
            grid_on (bool, optional): Whether to display grid lines. Defaults to False.

        Creates two subplots:
            1. Scatter plot of first two principal components
            2. Bar plot of the same data
        Both plots are color-coded by class labels.
        """
        if self.fail:
            print(f"Plotting sample process was stopped due to {self.errors}. Please check the logs for more information.")
            return

        x_vis = self.X_train[:sample_size]
        y_vis = self.y_train[:sample_size]

        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        dataset = tf.data.Dataset.from_tensor_slices((x_vis, y_vis))
        if augment:
            dataset = dataset.map(self._augment_tn_factory(), num_parallel_calls=tf.data.AUTOTUNE)

        # Collect the augmented dataset into numpy arrays
        X_augmented, y_augmented = [], []
        for x, y in dataset.as_numpy_iterator():
            X_augmented.append(x)
            y_augmented.append(y)

        X_augmented = np.array(X_augmented)
        y_augmented = np.array(y_augmented)

        # Flatten each sequence: shape becomes (n_samples, sequence_length * n_features)
        X_flat = X_augmented.reshape(X_augmented.shape[0], -1)

        # Normalize X
        X_flat = self._normalize_data(X_flat, method='std')

        # Reduce to 2D array
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_flat)

        # Label Legend and Plot
        classes = np.unique(y_augmented)
        colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))

        for i, class_id in enumerate(classes):
            idx = y_augmented == class_id
            label_name = self.encoder.inverse_transform([class_id])[0]
            ax1.scatter(X_2d[idx, 0], X_2d[idx, 1], label=label_name, color=colors[i], alpha=0.6, s=100, marker="*")
            ax2.bar(X_2d[idx, 0], X_2d[idx, 1], label=label_name, color=colors[i], alpha=0.6)

        if grid_on: plt.grid()

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

        if save_fig:
            try:
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f'{save_path}/pose_sample_plot.png', dpi=300)
            except Exception as e:
                print(f"Error: Could not create directory for saving figures: {e}")
                return

        plt.show()
        plt.close()


    @track_calls.trackcalls
    def get_tf_dataset(self, split="train", augment=False):
        """
        Create a TensorFlow dataset for the specified split.

        Args:
            split (str, optional): Which dataset split to use - "train", "val", or "test". Defaults to "train".
            augment (bool, optional): Whether to apply data augmentation. Defaults to False.

        Returns:
            tf.data.Dataset: TensorFlow dataset with the specified configuration.
        """

        if self.fail:
            print("An error occurred and dataset could not be created. Please check the logs for more information.")
            return
        if self.load_data.has_been_called and self.split_data.has_been_called:
            self.split = split
            if split == "val":
                return self._create_tf_dataset(self.X_valid, self.y_valid, augment=augment, shuffle=True)
            elif split == "test":
                return self._create_tf_dataset(self.X_test, self.y_test, augment=augment, shuffle=True)
            elif split == "train":
                return self._create_tf_dataset(self.X_train, self.y_train, augment=augment, shuffle=True)
            else:
                print(f"Error: Invalid split specified. Expected 'train', 'val', or 'test', got {split}")
                self.fail = True
                return None
        else:
            print(f"Error: Data was not loaded or splitted yet. Please call load_data() and split_data() first.")
            return None


    def save_pipeline(self, save_path, save_config=False):
        """
        Save the pipeline state and configuration to disk.

        Args:
            save_path (str): Path where to save the pipeline data.
            save_config (bool, optional): Whether to save configuration separately. Defaults to False.

        Notes:
            - Saves dataset element specification as pickle file
            - Saves TensorFlow dataset
            - Optionally saves configuration as JSON file
        """
        try:
            os.makedirs(save_path, exist_ok=True)

            with open(save_path + ".pickle", "wb") as file:
                pickle.dump(self.dataset.element_spec, file)

            tf.data.Dataset.save(self.dataset, save_path)

            if save_config:
                current_config = {
                    "data_dir": self.data_dir,
                    "sequence_length": self.sequence_length,
                    "landmarks_dim": self.landmarks_dim,
                    "batch_size": self.batch_size,
                    "augmentation_config": self.augmentation_config
                }

                config_path = os.path.join(save_path, "pipeline.config.json")
                with open(config_path, "w") as file:
                    json.dump(current_config, file, indent=4)

        except Exception as e:
            print(f"Error: Could not save pipeline: {e}")


    def load_pipeline(self, file_path, load_config=False, file_path_config=''):
        """
        Load a previously saved pipeline state.

        Args:
            file_path (str): Path to the saved pipeline data.
            load_config (bool, optional): Whether to load configuration file. Defaults to False.
            file_path_config (str, optional): Path to configuration file. Defaults to ''.

        Returns:
            dict: Pipeline configuration if load_config is True and successful.

        Notes:
            - Loads TensorFlow dataset
            - Optionally loads configuration from JSON file
        """
        self.dataset = tf.data.Dataset.load(file_path)

        if load_config:
            try:
                with open(file_path_config, 'r') as file:
                    return json.load(file)
            except FileNotFoundError:
                print(f"Error: File not found: {file_path_config}")
            except json.JSONDecodeError:
                print(f"Invalid JSON format in {file_path_config}")


    def _augment_tn_factory(self):
        """
        Create a function for applying data augmentations.

        Returns:
            callable: Function that applies configured augmentations to input data
                based on probability settings.

        Notes:
            - Internal method used by the pipeline
            - Supports noise, scale, and shift augmentations
            - Tracks augmentation application counts
        """
        def apply_aug(x, y):
            functions = {
                "noise": data_augmentation.adding_gaussian_noise,
                "scale": data_augmentation.adding_scaling,
                "shift": data_augmentation.adding_shifts
            }

            for fn, (enabled, probability) in self.augmentation_config.items():
                if enabled:
                    tf.print(f"Applying augmentation: {fn}")

                    # Create a random condition for applying augmentation
                    should_apply = tf.random.uniform([], 0, 1) < probability
                    aug_fn = functions[fn](probability)

                    def no_change():
                        return x, y

                    x, y = tf.cond(should_apply, lambda: aug_fn(x, y), no_change)
                    self.augmentation_counts[fn] += 1

            return x, y

        return apply_aug


    def _create_tf_dataset(self, X: np.ndarray, y: np.ndarray, augment: bool = False, shuffle: bool = True):
        """
        Create a TensorFlow dataset from input arrays.

        Args:
            X (np.ndarray): Input features array.
            y (np.ndarray): Labels array.
            augment (bool, optional): Whether to apply augmentation. Defaults to False.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.

        Returns:
            tf.data.Dataset: Configured TensorFlow dataset.

        Notes:
            - Internal method used by get_tf_dataset
            - Applies caching for training data
            - Configures batching and prefetching
        """
        dataset = tf.data.Dataset.from_tensor_slices((X, y))

        if augment:
            dataset = dataset.map(self._augment_tn_factory(), num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))

        if self.split == "train":
            self.dataset = dataset.cache().batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        else:
            self.dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return self.dataset


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
            min_val = tf.reduce_min(data)
            max_val = tf.reduce_max(data)
            scaled_data = (data - min_val) / (max_val - min_val)
            return scaled_data

        if method == 'std':
            mean_val = tf.reduce_mean(data)
            std_val = tf.math.reduce_std(data)
            scaled_data = (data - mean_val) / std_val
            return scaled_data

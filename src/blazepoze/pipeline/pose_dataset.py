# Standard libraries
import os
import pickle
import textwrap
import json
import logging
from collections import defaultdict

# Third-party libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from pandas.errors import EmptyDataError
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Local libraries
from src.blazepoze.utils import logging_utils
from src.blazepoze.utils import augment, validation

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PoseDatasetPipeline():
    """
    PoseDatasetPipeline

    A pipeline for processing, augmenting, and preparing pose estimation data for machine learning models.

    This class provides a comprehensive pipeline for handling pose estimation datasets, including
    data loading, preprocessing, augmentation, and conversion to TensorFlow datasets. It supports
    various data augmentation techniques and provides utilities for saving/loading pipeline states.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the dataset files.
    sequence_length : int
        Number of frames in each sequence (expected to be 50).
    landmarks_dim : int
        Number of landmark dimensions in the input data (expected to be 99).
    batch_size : int
        Size of batches for training and validation.
    augmentation_config : dict
        Configuration for data augmentation with format:
        {'augmentation_type': [bool, probability]}
        where augmentation_type can be 'noise', 'scale', or 'shift',
        bool indicates if enabled, and probability is a float in range [0-1].

    Attributes
    ----------
    X : ndarray
        Loaded pose sequences of shape (n_samples, sequence_length, landmarks_dim).
    y : ndarray
        Original class labels.
    y_encoder : ndarray
        Encoded class labels.
    encoder : LabelEncoder
        Scikit-learn label encoder for converting between string and integer labels.
    X_train : ndarray
        Training data after splitting.
    X_valid : ndarray
        Validation data after splitting.
    X_test : ndarray
        Test data after splitting.
    y_train : ndarray
        Training labels after splitting.
    y_valid : ndarray
        Validation labels after splitting.
    y_test : ndarray
        Test labels after splitting.
    data_frames_list : list
        List of individual pose data frames.
    augmentation_counts : defaultdict
        Counter for tracking applied augmentations.
    fail : bool
        Flag indicating if any operation has failed.
    errors : list
        List of encountered errors.

    Methods
    -------
    load_data()
        Load and preprocess pose data from the specified directory.
    split_data(test_size=0.2, valid_size=0.1)
        Split loaded data into training, validation, and test sets.
    get_tf_dataset(split="train", augment=False)
        Create a TensorFlow dataset for the specified split.
    save_pipeline(save_path, save_config=False)
        Save pipeline state and configuration to disk.
    load_pipeline(file_path, load_config=False, file_path_config='')
        Load a previously saved pipeline state.

    Properties
    ----------
    class_names : ndarray
        Unique class names in the dataset.
    class_names_encoded : int
        Number of unique classes in the dataset.

    Notes
    -----
    - The pipeline expects CSV files with specific naming patterns (_landmarks.csv)
    - Data augmentation is applied only when explicitly requested
    - The pipeline maintains data distribution through stratified splitting
    - All operations are tracked for debugging purposes

    Raises
    ------
    Exception
        If augmentation.json format is invalid
    AssertionError
        If loaded data doesn't match expected dimensions
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
            Exception: If augmentation.json format is invalid.
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.landmarks_dim = landmarks_dim
        self.batch_size = batch_size
        self.augmentation_counts = defaultdict(int)
        self.fail = False
        self.errors = []

        if validation.check_instance(augmentation_config):
            self.augmentation_config = augmentation_config
        else:
            raise Exception("""An error occurred when trying to configure augmentation specifications.
            Make sure the attribute augmentation.json follows the following format:
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
                Gesture Classes: {np.unique(self.y)}
                Augmentation Counts: {self.augmentation_counts}
            """)
        else:
            loaded_data = "Data was not loaded yet. No information about data is available!\n\n"

        load_check = "✅" if self.load_data.has_been_called else "❌"
        split_check = "✅" if self.split_data.has_been_called else "❌"
        dataset_check = "✅" if self.get_tf_dataset.has_been_called else "❌"

        description_line = f"PoseDatasetPipeline (Loaded: {load_check} | Split: {split_check} | Dataset Created: {dataset_check})"

        return relevant_metadata + loaded_data + description_line


    @logging_utils.trackcalls
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
        file_config = ("d__landmarks.csv", "p__landmarks.csv", "ft__landmarks.csv", "ht__landmarks.csv", "alt__landmarks.csv")
        self.data_frames_list = []

        try:
            for folder in os.listdir(self.data_dir):  # Navigate each subfolder
                folder_path = os.path.join(self.data_dir, folder)

                if not os.path.isdir(folder_path): continue  # Skip if it's not a folder

                for file in os.listdir(folder_path):  # Navigate each folder per subfolder
                    file_path = os.path.join(folder_path, file)
                    if (file.endswith(file_config)):
                        pose_df = pd.read_csv(file_path)

                        if pose_df.shape != (self.sequence_length, self.landmarks_dim + 1):
                            continue  # +1 for 'Unnamed: 0'

                        self.frame_data = pose_df.drop(columns=pose_df.columns[0]).values.astype(np.float32)  # Shape: (50, 99)
                        self.data_frames_list.append(self.frame_data)

                        X_data.append(self.frame_data)

                        if file.endswith("alt__landmarks.csv"):
                            label = file.split("_alt__")[0]
                            y_labels.append(label)
                        else:
                            label = file.split("__")[0]
                            y_labels.append(label)

            # Transform data frames to np arrays
            self.X = np.array(X_data, dtype=np.float32)  # Shape: (samples, 50, 99)
            self.y = np.array(y_labels)

            # Encoder labels
            self.encoder = LabelEncoder()
            self.y_encoder = self.encoder.fit_transform(self.y)

            assert self.X.shape[1:] == (50, 99), "X does not satisfy the (50, 99) shape"

        except FileNotFoundError:
            logger.warning(f"File not found")
            self.fail = True
            self.errors.append("File not found")
        except EmptyDataError:
            logger.warning(f"Error: Empty data in file")
            self.fail = True
            self.errors.append("Empty data in file")
        except pd.errors.ParserError as e:
            logger.warning(f"Error: Parser error: {e}")
            self.fail = True
            self.errors.append(f"Parser error: {e}")
        except Exception as e:
            logger.warning(f"Error: Unknown error occurred: {e}")
            self.fail = True
            self.errors.append(f"Unknown error occurred: {e}")


    @logging_utils.trackcalls
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
            logger.warning(f"Splitting data process was stopped due to {self.errors}. Please check the logs for more information.")
            return
        counts = np.bincount(self.y_encoder)
        stratify = self.y_encoder if np.min(counts) >= 2 else None

        X_train_full, self.X_test, y_train_full, self.y_test = train_test_split(
            self.X, self.y_encoder, test_size=test_size, stratify=stratify
        )

        if len(y_train_full) > 1:
            counts_train = np.bincount(y_train_full)
            stratify_val = y_train_full if np.min(counts_train) >= 2 else None
            self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
                X_train_full, y_train_full, test_size=valid_size, stratify=stratify_val
            )
        else:
            self.X_train = X_train_full
            self.X_valid = np.empty((0, self.sequence_length, self.landmarks_dim))
            self.y_train = y_train_full
            self.y_valid = np.empty((0,), dtype=y_train_full.dtype)

    @logging_utils.trackcalls
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
            logger.warning("An error occurred and dataset could not be created. Please check the logs for more information.")
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
                logger.warning(f"Error: Invalid split specified. Expected 'train', 'val', or 'test', got {split}")
                self.fail = True
                return None
        else:
            logger.warning(f"Error: Data was not loaded or split yet. Please call load_data() and split_data() first.")
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
                    "augmentation.json": self.augmentation_config
                }

                config_path = os.path.join(save_path, "pipeline.config.json")
                with open(config_path, "w") as file:
                    json.dump(current_config, file, indent=4)

        except Exception as e:
            logger.warning(f"Error: Could not save pipeline: {e}")


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
                logger.warning(f"Error: File not found: {file_path_config}")
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON format in {file_path_config}")


    @property
    def class_names(self):
        return np.unique(self.y)


    @property
    def class_names_encoded(self):
        return len(np.unique(self.y_encoder))


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
                "noise": augment.adding_gaussian_noise,
                "scale": augment.adding_scaling,
                "shift": augment.adding_shifts
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

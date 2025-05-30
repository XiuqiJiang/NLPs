import numpy as np
import umap
import matplotlib.pyplot as plt
import glob
import sys
import os
import argparse
import torch

# Add the project root directory to the Python path
# Assumes this script is in NLPs/src/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Import necessary modules from your project structure
try:
    # Import ESMVAEToken from the correct location
    from src.models.vae_token import ESMVAEToken
    # Import create_data_loaders to load the dataset
    from src.utils.data_utils import create_data_loaders
    # Import config relative to the project root
    from NLPs.config import config
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure that your project structure is correct and necessary files (vae_token.py, data_utils.py, config.py) are in place.")
    sys.exit(1)

# Remove the placeholder Dataset class
# We will use create_data_loaders instead
# class ProteinDataset(torch.utils.data.Dataset):
#    ...


def load_model(model_path: str, config: dict) -> ESMVAEToken | None: # Add type hints
    if ESMVAEToken is None:
        print("Model class not available due to import error.")
        return None
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    # 支持 checkpoint 格式和直接 state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model = ESMVAEToken(**config)
    model.load_state_dict(state_dict)
    return model

def main():
    parser = argparse.ArgumentParser(description='Visualize VAE Latent Space')
    parser.add_argument('--model_path', type=str, required=False, help=f'Path to the trained model file (.pth). If not provided, the latest .pth file in {{config.MODEL_SAVE_DIR}} will be used.')
    parser.add_argument('--data_path', type=str, default=config.EMBEDDING_FILE, help='Path to the processed dataset file (.pt)')
    parser.add_argument('--output_dir', type=str, default='./visualization_output', help='Directory to save visualization data')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Resolve model path (Keep existing logic)
    model_path = args.model_path
    if not model_path:
        default_model_dir = config.MODEL_SAVE_DIR
        print(f"Model path not provided. Looking for the latest model in {default_model_dir}")
        try:
            list_of_files = glob.glob(os.path.join(default_model_dir, '*.pth'))
            if not list_of_files:
                print(f"Error: No .pth model files found in {default_model_dir}")
                return

            latest_file = max(list_of_files, key=os.path.getmtime)
            model_path = latest_file
            print(f"Using latest model file: {model_path}")
        except Exception as e:
            print(f"Error finding latest model file: {e}")
            print("Please provide the model path manually using --model_path.")
            return
    else:
        if not os.path.exists(model_path):
            print(f"Error: Specified model file not found at {model_path}")
            return
        print(f"Using specified model file: {model_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Define model configuration based on config.py
    # Ensure all necessary parameters for ESMVAEToken are included and correctly named
    model_config = {
        'input_dim': getattr(config, 'ESM_EMBEDDING_DIM', 768),
        'hidden_dims': getattr(config, 'HIDDEN_DIMS', [512, 256]),
        'latent_dim': getattr(config, 'LATENT_DIM', 256),
        'vocab_size': getattr(config, 'VOCAB_SIZE', 33),
        'max_sequence_length': getattr(config, 'MAX_SEQUENCE_LENGTH', 64),
        'pad_token_id': getattr(config, 'PAD_TOKEN_ID', 0),
        'use_layer_norm': getattr(config, 'USE_LAYER_NORM', True),
        'dropout': getattr(config, 'DROPOUT', 0.1),
        'rnn_hidden_dim': getattr(config, 'RNN_HIDDEN_DIM', 64),
        'num_rnn_layers': getattr(config, 'NUM_RNN_LAYERS', 1),
        'ring_embedding_dim': getattr(config, 'RING_EMBEDDING_DIM', 32),
        'num_classes': 3,  # 只支持3C,4C,5C
    }

    # IMPORTANT: Get vocab_size, max_sequence_length, and pad_token_id from the actual data/tokenizer if possible.
    # For now, relying on config defaults or placeholders. This might be a source of error if config is incomplete.
    # A more robust way is to load the tokenizer used during training and get these values from it.
    # Let's assume for now config has these or the defaults are correct.

    model = load_model(model_path, model_config) # Pass config
    if model is None:
        return
    model.to(device)
    model.eval()

    # Load the dataset using create_data_loaders
    # Assuming create_data_loaders returns train_loader, val_loader
    # We need the dataset object itself to iterate through all data points for visualization
    # Let's modify this to load the full dataset or use a combined dataloader
    # create_data_loaders might not be suitable if it only provides train/val split with DataLoader batching.
    # Let's revert to loading the dataset directly using the placeholder class for now
    # and ensure the placeholder class can load the .pt file correctly.

    # Reverting data loading to use the placeholder ProteinDataset
    # We need to ensure the data file at args.data_path contains the necessary data in the expected format.
    # Based on test_reconstruction.py, the data loader provides 'input_ids' and 'embeddings'.
    # The placeholder ProteinDataset was assumed to provide 'input_ids' and 'ring_info'.
    # We need 'input_ids' (or embeddings) and 'ring_info' for visualization.
    # Let's assume the .pt file at args.data_path contains a structure that, when loaded
    # and iterated, provides batches with 'input_ids' and 'ring_info'.
    # If create_data_loaders returns a dataset object, we could use that.

    # Let's re-examine test_reconstruction.py data loading
    # It uses create_data_loaders which takes EMBEDDING_FILE and returns train/val loaders.
    # The batches from these loaders contain 'input_ids' and 'embeddings'.
    # Our visualization needs 'input_ids' (to pass to model if needed, though we use embeddings for mu) and 'ring_info'.
    # The ring_info must be available in the dataset loaded by create_data_loaders.

    # Let's try using create_data_loaders for the validation set, as it should contain ring_info
    # We need to ensure create_data_loaders provides ring_info in its batches.
    # Reviewing test_reconstruction.py again, it seems the batch contains 'input_ids', 'embeddings', and possibly others.
    # The loss calculation in training/validation would need ring_info if ring prediction is involved.
    # Let's assume create_data_loaders batches include 'ring_info'.

    print(f"Loading dataset from {args.data_path} using create_data_loaders...")
    # create_data_loaders takes data_file, batch_size, train_test_split, etc.
    # We need the full dataset for visualization, not just train/val split with DataLoader batching.
    # create_data_loaders returns DataLoaders, not the raw dataset.
    # We need to adapt create_data_loaders or manually load the dataset file.

    # Option 1: Adapt create_data_loaders to return the full dataset object.
    # Option 2: Manually load the .pt file and create a single DataLoader for the full dataset.

    # Let's choose Option 2 for now, as it's simpler for this script.
    # We need to replicate the dataset loading part from create_data_loaders or a similar script.
    # Assuming args.data_path points to a .pt file containing data suitable for DataLoader.
    # Let's assume it loads a list of dicts, where each dict is a data point.

    # Manual Dataset Loading (revisiting placeholder idea)
    # This requires the .pt file structure to be known.
    # If the .pt file contains a list of samples, each like {'input_ids': tensor, 'ring_info': int}
    # then the placeholder ProteinDataset would work if its loading logic is correct.

    # Let's assume the .pt file is a list of dictionaries, each dictionary representing a sample.
    # We need to load this list and create a DataLoader.

    print(f"Manually loading dataset from {args.data_path}...")
    try:
        # Assuming the .pt file contains the data directly loadable by torch.load
        dataset_data = torch.load(args.data_path)
        print(f"Loaded {len(dataset_data)} data points.")

        # Create a simple Dataset from the loaded data
        # This assumes dataset_data is iterable and each item is a sample
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                if isinstance(self.data, dict):
                    key = list(self.data.keys())[idx]
                    item = self.data[key]
                else:
                    item = self.data[idx]
                # 自动将所有 list 字段转为 tensor
                for k, v in item.items():
                    if isinstance(v, list):
                        item[k] = torch.tensor(v)
                return item

        dataset = SimpleDataset(dataset_data)

        # Create a DataLoader for the full dataset
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64, # Use a reasonable batch size
            shuffle=False,
            num_workers=0, # Adjust as needed
            pin_memory=True if args.device == 'cuda' else False,
        )

    except FileNotFoundError:
        print(f"Error: Dataset file not found at {args.data_path}")
        return
    except Exception as e:
        print(f"Error loading or processing dataset: {e}")
        return


    # --- Data extraction loop (Keep existing logic) ---
    latent_vectors = []
    ring_infos = []

    print("Extracting latent vectors and ring info...")
    with torch.no_grad():
        for batch in dataloader:
            # Ensure batch contains 'input_ids' or 'embeddings' and 'ring_info'
            # Based on test_reconstruction.py, batches from create_data_loaders have 'embeddings'.
            # Our model.encode expects embeddings.
            # And we need 'ring_info' for coloring.
            if 'embeddings' not in batch or 'ring_info' not in batch:
                print("Error: Dataset batch does not contain 'embeddings' or 'ring_info'.")
                print("Please check the dataset file structure or the data loading process.")
                return

            embeddings = batch['embeddings'].to(device)
            ring_info = batch['ring_info'].to(device) # Keep on device for potential model use if needed, but move to CPU for numpy later

            try:
                # Get mu from the encoder using embeddings
                mu, logvar = model.encode(embeddings)

                latent_vectors.append(mu.cpu().numpy())
                # Get original ring_info (on CPU) for plotting
                ring_infos.append(batch['ring_info'].numpy())

            except Exception as e:
                print(f"Error processing batch: {e}")
                # Depending on the error, you might want to skip the batch or stop
                continue # Skip this batch

    # --- UMAP and Plotting (Keep existing logic) ---
    if latent_vectors and ring_infos:
        all_latent_vectors = np.concatenate(latent_vectors, axis=0)
        all_ring_infos = np.concatenate(ring_infos, axis=0)

        print(f"Extracted {len(all_latent_vectors)} latent vectors.")
        print(f"Corresponding ring info shape: {all_ring_infos.shape}")

        latent_vectors_path = os.path.join(args.output_dir, 'latent_vectors.npy')
        ring_infos_path = os.path.join(args.output_dir, 'ring_infos.npy')
        
        np.save(latent_vectors_path, all_latent_vectors)
        np.save(ring_infos_path, all_ring_infos)

        print(f"Saved latent vectors to {latent_vectors_path}")
        print(f"Saved ring info to {ring_infos_path}")

        print("\nPerforming UMAP dimensionality reduction...")
        n_neighbors = min(len(all_latent_vectors) - 1, 15)
        if n_neighbors <= 1:
            print("Not enough data points for UMAP with n_neighbors > 1. Skipping UMAP.")
        else:
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
            try:
                embedding = reducer.fit_transform(all_latent_vectors.astype(np.float32))

                print("\nCreating visualization plot...")
                plt.figure(figsize=(10, 8))

                unique_rings = np.unique(all_ring_infos)
                unique_rings.sort()

                cmap = plt.cm.get_cmap('viridis', len(unique_rings))
                norm = plt.Normalize(vmin=unique_rings.min(), vmax=unique_rings.max())

                scatter = plt.scatter(
                    embedding[:, 0],
                    embedding[:, 1],
                    c=all_ring_infos,
                    cmap=cmap,
                    norm=norm,
                    s=10,
                    alpha=0.7
                )

                cbar = plt.colorbar(scatter)
                cbar.set_label('Cysteine Ring Count')
                cbar.set_ticks(unique_rings)
                cbar.set_ticklabels(unique_rings)

                plt.title('UMAP projection of VAE Latent Space (colored by Cysteine Ring Count)')
                plt.xlabel('UMAP Component 1')
                plt.ylabel('UMAP Component 2')
                plt.grid(True, linestyle='--', alpha=0.5)

                plot_path = os.path.join(args.output_dir, 'latent_space_umap_2d.png')
                plt.savefig(plot_path)
                print(f"Saved UMAP plot to {plot_path}")

            except Exception as e:
                 print(f"Error during UMAP or plotting: {e}")
                 print("Please ensure umap-learn and matplotlib are installed.")

    else:
        print("No data extracted or loaded for visualization.")

if __name__ == '__main__':
    main() 
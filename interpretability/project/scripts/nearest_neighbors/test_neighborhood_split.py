import polars as pl
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from project.utils.strs import SEED
from project.utils.functions import load_config, get_cached_embeddings, get_distance_matrices
from project.utils.metrics import knn_metrics

def get_neighbor_model(dataset_name, type = 'k', weights = 'distance'):
	dataset_config = load_config(dataset_name)
	if 'classification' in dataset_config['task_type']:
		if type == 'k':
			from sklearn.neighbors import KNeighborsClassifier
			model = KNeighborsClassifier(
				weights = weights,
				metric='precomputed',
			)
		elif type ==  'radius':
			from sklearn.neighbors import RadiusNeighborsClassifier
			model = RadiusNeighborsClassifier(
				weights=weights,
				metric='precomputed',
				outlier_label='most_frequent'
			)
	elif 'regression' in dataset_config['task_type']:
		if type == 'k':
			from sklearn.neighbors import KNeighborsRegressor
			model = KNeighborsRegressor(
				weights = weights,
				metric='precomputed',
			)
		elif type == 'radius':
			from sklearn.neighbors import RadiusNeighborsRegressor
			model = RadiusNeighborsRegressor(
				weights= weights,
				metric='precomputed'
			)
	return model

def run_neighborhood_probe(cfg: DictConfig):
    
    rng = np.random.default_rng(seed = cfg.data.seed)

    # Use parameters from the configuration
    model_shorthand = cfg.plm_model.shorthand
    model_layer_num = cfg.plm_model.model_layer_num
    dataset_name = cfg.data.dataset_name
    num_folds = cfg.data.num_folds
    subsample_fraction = cfg.data.subsample_fraction
    neighbor_ks = cfg.knn.neighbor_ks
    flavour_of_nn = cfg.knn.nn_flavour
    neighbor_weight = cfg.knn.neighbor_weight
    distance_metric = cfg.knn.distance_metric

    # Hydra handles paths relative to the execution directory
    knn_dir = Path(cfg.paths.save_dir)
    knn_dir.mkdir(parents=True, exist_ok=True)
    # knn_fig_dir = knn_dir / 'figures'
    # knn_fig_dir.mkdir(parents=True, exist_ok=True)
    # knn_preds_dir = knn_dir / 'predictions'
    # knn_preds_dir.mkdir(parents=True, exist_ok=True)    
    
    print(f"Loading data")

    results = []
    results_file = knn_dir / f"{dataset_name}_{model_shorthand}_{model_layer_num}_{flavour_of_nn}_{distance_metric}_neighbors_multilabel_results_split.parquet.gz"

    # Load dataset and get sequences and labels
    df = pl.read_parquet(load_config(dataset_name)['data_path'])

    # Get sequences and labels
    sequences = df['sequence']
    label_cols = [col for col in df.columns if (df[col].dtype == pl.Boolean)]
    labels = df[label_cols].to_numpy()

    test_inds = df.with_row_index(name='row_index').filter(pl.col('split') == 'test')['row_index'].to_list()
    train_inds = df.with_row_index(name='row_index').filter(pl.col('split') == 'train')['row_index'].to_list()

    train_labels = labels[train_inds]
    test_labels = labels[test_inds]

    # Get pre-cached embeddings for the sequences at the target layer
    layer_embeddings = get_cached_embeddings(sequences = sequences,
                                            model_shorthand=model_shorthand, 
                                            layer_nums = [model_layer_num])[model_layer_num]

    # Precalculate distance matrices because we want to do splitting and don't want to constantly recalc the same thing
    embedding_treatments = get_distance_matrices(layer_embeddings, distance_metric, SEED) # returns a dict of 'embeddings', 'shuffled_embeddings', 'shuffled_positions': dmatrix

    # For actual embeddings and control embeddings
    for treatment, matrix_dict in embedding_treatments.items():

        # Fetch the distance matrix
        dmatrix = matrix_dict['matrix']
    
        # Train the knn on the training data
        train_train_distances = dmatrix[train_inds, :][:, train_inds]
        test_train_distances = dmatrix[test_inds, :][:, train_inds]

        # Make KNN - naturally multilabel
        nn_model = get_neighbor_model(dataset_name, type= flavour_of_nn, weights=neighbor_weight)

        # Fit on the training data
        nn_model.fit(train_train_distances, train_labels)

        test_predictions = {}
        test_probabilities = {}

        for k in neighbor_ks:
            # Set the number of neighbors
            nn_model.n_neighbors = k

            # Make predictions for test points
            predicted_labels = nn_model.predict(test_train_distances)
            
            if hasattr(nn_model, 'predict_proba'):
                probs_positive_class = nn_model.predict_proba(test_train_distances)
            else:
                # If it's a regression model or the classification model doesn't have predict_proba
                # We calculate scores based on neighbors manually (using k-neighbors search)
                distances, indices = nn_model.kneighbors(test_train_distances, n_neighbors=k, return_distance=True)
                neighbor_labels = train_labels[indices] # Labels of the k-neighbors
                
                if neighbor_weight == 'distance':
                    # Score = Sum(Inverse Distance * Neighbor Label) / Sum(Inverse Distance)
                    weights = 1.0 / (distances + 1e-10) 
                    weights_sum = np.sum(weights, axis=1, keepdims=True)
                    # Expand weights to match the label shape (samples, k, labels)
                    weights_expanded = np.repeat(weights[:, :, np.newaxis], neighbor_labels.shape[-1], axis=2)
                    weighted_label_sum = np.sum(weights_expanded * neighbor_labels, axis=1)
                    probs_positive_class = weighted_label_sum / weights_sum 
                else: # 'uniform'
                    probs_positive_class = np.mean(neighbor_labels, axis=1) # Mean label across neighbors

            test_predictions[k] = predicted_labels
            test_probabilities[k] = probs_positive_class
                
        # The loop iterates `num_folds` times to perform subsampling of the test set for robust scoring.
        for split_num in range(num_folds):
            
            # Randomly subsample the indices of the TEST set
            # The size is subsample_fraction (assumed to be <= len(test_inds))
            subsample_indices_relative = rng.choice(len(test_inds), size = int(len(test_inds) * subsample_fraction), replace=False)
            # Subsample the target labels (Test Set)

            targets_subsample = np.array(test_labels)[subsample_indices_relative.flatten().tolist()]

            for k in neighbor_ks:
                # Subsample the predictions and probabilities corresponding to k
                predictions_subsample = np.array(test_predictions[k])[subsample_indices_relative.flatten().tolist()]

                probabilities_list = test_probabilities[k]           
                
                probabilities_list = np.vstack([i[:,-1] for i in probabilities_list]).T

                probabilities_subsample = probabilities_list[subsample_indices_relative] # Get indices we want

                result_info = {
                        'dataset': dataset_name,
                        'model_name': model_shorthand,
                        'layer_num': model_layer_num,
                        'control_type': treatment,
                        'k': k,
                        'distance_metric': distance_metric,
                        'weight_strategy': neighbor_weight,
                        'fold': split_num,
                        'split_strategy': 'linear_probing',
                        }
                # Score the predictions
                for metric_name, metric_func in knn_metrics.items():
                    # These metrics require predicted probabilities
                    if metric_name in ['coverage_error', 'label_ranking_average_precision_score', 'label_ranking_loss', 'auroc']:
                        print(targets_subsample.shape, probabilities_subsample.shape)
                        result_info[metric_name] = metric_func(targets_subsample, probabilities_subsample)
                    else:
                        result_info[metric_name] = metric_func(targets_subsample, predictions_subsample)

                # Score results
                results.append(result_info)
                    
    pl.DataFrame(results).write_parquet(results_file)

    print('script complete')

# --- HYDRA ENTRY POINT ---
@hydra.main(config_path="conf", config_name="default", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    run_neighborhood_probe(cfg)

if __name__ == "__main__":
    main()
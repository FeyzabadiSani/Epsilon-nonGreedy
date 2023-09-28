import logging
from models.r3Models import initialize_models
from runner.common import train
from util.evaluation import initialize_metrics, update_metrics
from DSG.common import initialize_models_training_dataset, update_datasets
from util.common import DatasetName

def run(logger: logging.Logger, config: dict, data: dict):
    # Initialize Metrics
    auc, logloss = initialize_metrics(
        models_spec=config['models_spec'],
        n_experiments=config['n_experiments'],
        n_batches=config['n_batches']
        )
    
    for exp_idx in range(config['n_experiments']):
        logger.error(r"Experiment %d / %d starts." % (exp_idx + 1, config['n_experiments']))

        # Foreach model we use different training dataset.
        datasets = initialize_models_training_dataset(config['models_spec'])

        for batch_idx in range(config['n_batches']):
            logger.error(r"     Processing batch %d / %d." % (batch_idx + 1, config['n_batches']))

            # Models retrained in each training round!
            models = initialize_models(config['models_spec'])
            logger.error(r"     Models have been initialized.")

            # Each model has it's own training dataset
            update_datasets(
                models=models,
                datasets=datasets,
                path=config['path'],
                pick_ratio=config['pick_ratio'],
                uniform_batch=data['train']['uniform'][batch_idx],
                biased_batch=data['train']['biased'][batch_idx],
                dataset_name=DatasetName.yahooR3,
            )
            logger.error("     Datasets have been updated.")

            # Train models
            train(
                models=models,
                datasets=datasets,
                unobserved_data=None,
                validation_data=data['st_validation'],
                logger=logger,
                n_epochs=config['n_epochs'],
                saving_path=config['path'],
                conv_wait_steps=config['conv_wait_steps'],
                dataset_name=DatasetName.yahooR3,
                )
            
            #Evaluate 
            update_metrics(
                models=models,
                eval_data=data['st_test'], # TODO Change for test
                experiment_idx=exp_idx,
                batch_idx=batch_idx,
                auc=auc, 
                logloss=logloss,
                saving_path=config['path'],
                dataset_name=DatasetName.yahooR3,
            )
    return auc, logloss
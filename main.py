import hydra
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig

from dataset.dataset import TimeSeriesDataModule
from utils.main_utils.logging_utils import (
    post_setup_wandb_logger,
    setup_logging,
    setup_wandb_logger,
    write_wandb_summary,
)
from utils.main_utils.run_context import build_run_context
from utils.main_utils.train_test_utils import (
    build_callbacks,
    build_trainer,
    run_testing,
    run_training,
    setup_environment,
)


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Build run context + output directory path
    run_context = build_run_context(cfg)
    output_dir = str(run_context.ensure())

    # Setup
    setup_logging(output_dir)
    setup_environment(cfg)

    # Model
    model = instantiate(cfg.model.module, _convert_="partial")

    # Logger + callbacks + trainer
    wandb_logger = setup_wandb_logger(cfg, output_dir, model)
    callbacks, ckpt_cb = build_callbacks(cfg, output_dir)
    trainer = build_trainer(cfg, wandb_logger, callbacks)

    if trainer.global_rank == 0:
        post_setup_wandb_logger(wandb_logger, cfg, output_dir, model)
    

    # Data
    if cfg.dataset.source == "fev":
        from dataset.dataset_fev import FevDataModule
        dm = FevDataModule(cfg)
    else:
        dm = TimeSeriesDataModule(cfg)

    # Train
    run_training(trainer, model, dm, cfg)

    # Clean up DDP group if it exists to allow clean single-GPU execution or exit
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    # Test
    # Run testing only on rank 0 and with single GPU to avoid DDP dataset splitting issues
    if trainer.global_rank == 0:
        if cfg.gpus > 1:
            # Create a localized config for testing
            # Use 'copy' to avoid affecting other parts if they existed
            test_cfg = cfg.copy()
            test_cfg.gpus = 1
            test_cfg.strategy = "auto"
            
            # Build a new trainer for testing
            test_trainer = build_trainer(test_cfg, wandb_logger, callbacks)
            run_testing(test_trainer, model, dm, cfg, ckpt_cb, run_context)
        else:
            run_testing(trainer, model, dm, cfg, ckpt_cb, run_context)
        write_wandb_summary(wandb_logger)


if __name__ == "__main__":
    main()

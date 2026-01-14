"""
Weights & Biases experiment tracker.

Provides integration with WandB for experiment tracking and visualization.
"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class WandBTracker:
    """
    Weights & Biases experiment tracker.

    Handles initialization, logging, and finalization of WandB runs.
    """

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize WandB tracker

        Args:
            project: WandB project name
            entity: WandB entity (team) name
            tags: List of tags for the run
            notes: Notes about the run
            config: Initial configuration dictionary
            **kwargs: Additional WandB init arguments
        """
        self.project = project
        self.entity = entity
        self.tags = tags or []
        self.notes = notes
        self.config = config or {}
        self.kwargs = kwargs
        self.run = None

        # Check if wandb is installed
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            raise ImportError(
                "wandb is not installed. Please install with: pip install wandb"
            )

    def init_run(
        self,
        config: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Initialize WandB run

        Args:
            config: Configuration dictionary to log
            run_name: Optional name for the run
            **kwargs: Additional arguments for wandb.init
        """
        logger.info(f"Initializing WandB run in project: {self.project}")

        # Merge configs
        full_config = {**self.config, **(config or {})}

        # Initialize run
        self.run = self.wandb.init(
            project=self.project,
            entity=self.entity,
            name=run_name,
            tags=self.tags,
            notes=self.notes,
            config=full_config,
            **{**self.kwargs, **kwargs}
        )

        logger.info(f"WandB run initialized: {self.run.name}")
        logger.info(f"  URL: {self.run.url}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        """
        Log metrics to WandB

        Args:
            metrics: Dictionary of metric name to value
            step: Optional step number
            commit: Whether to commit the log
        """
        if self.run is None:
            logger.warning("WandB run not initialized, call init_run() first")
            return

        self.wandb.log(metrics, step=step, commit=commit)

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to WandB config

        Args:
            params: Dictionary of parameters
        """
        if self.run is None:
            logger.warning("WandB run not initialized, call init_run() first")
            return

        self.wandb.config.update(params)
        logger.info(f"Logged {len(params)} parameters to WandB")

    def log_artifact(
        self,
        path: str,
        name: Optional[str] = None,
        artifact_type: str = "model",
        aliases: Optional[List[str]] = None,
    ) -> None:
        """
        Log artifact to WandB

        Args:
            path: Path to artifact (file or directory)
            name: Artifact name (defaults to path basename)
            artifact_type: Type of artifact
            aliases: List of aliases for the artifact
        """
        if self.run is None:
            logger.warning("WandB run not initialized, call init_run() first")
            return

        from pathlib import Path
        path_obj = Path(path)

        if name is None:
            name = path_obj.name

        logger.info(f"Logging artifact: {name}")

        artifact = self.wandb.Artifact(name=name, type=artifact_type)

        if path_obj.is_dir():
            artifact.add_dir(str(path))
        elif path_obj.is_file():
            artifact.add_file(str(path))
        else:
            raise ValueError(f"Path does not exist: {path}")

        self.run.log_artifact(artifact, aliases=aliases)
        logger.info("Artifact logged successfully")

    def log_table(
        self,
        name: str,
        data: List[List[Any]],
        columns: List[str],
    ) -> None:
        """
        Log table to WandB

        Args:
            name: Table name
            data: Table data as list of rows
            columns: Column names
        """
        if self.run is None:
            logger.warning("WandB run not initialized, call init_run() first")
            return

        table = self.wandb.Table(columns=columns, data=data)
        self.wandb.log({name: table})
        logger.info(f"Logged table: {name}")

    def log_image(
        self,
        name: str,
        image,
        caption: Optional[str] = None,
    ) -> None:
        """
        Log image to WandB

        Args:
            name: Image name
            image: Image (PIL Image, numpy array, or path)
            caption: Optional caption
        """
        if self.run is None:
            logger.warning("WandB run not initialized, call init_run() first")
            return

        wandb_image = self.wandb.Image(image, caption=caption)
        self.wandb.log({name: wandb_image})

    def watch_model(
        self,
        model,
        log: str = "gradients",
        log_freq: int = 100,
    ) -> None:
        """
        Watch model for gradient and parameter tracking

        Args:
            model: PyTorch model to watch
            log: What to log ("gradients", "parameters", "all")
            log_freq: Logging frequency
        """
        if self.run is None:
            logger.warning("WandB run not initialized, call init_run() first")
            return

        self.wandb.watch(model, log=log, log_freq=log_freq)
        logger.info(f"Watching model with log={log}, freq={log_freq}")

    def finish(self) -> None:
        """Finish WandB run"""
        if self.run:
            logger.info("Finishing WandB run")
            self.run.finish()
            self.run = None
        else:
            logger.warning("No active WandB run to finish")

    def get_callback(self):
        """
        Get WandB callback for HuggingFace Trainer

        Returns:
            WandbCallback instance
        """
        try:
            from transformers.integrations import WandbCallback
            return WandbCallback()
        except ImportError:
            logger.warning("WandbCallback not available in transformers")
            return None

    @property
    def run_id(self) -> Optional[str]:
        """Get current run ID"""
        return self.run.id if self.run else None

    @property
    def run_name(self) -> Optional[str]:
        """Get current run name"""
        return self.run.name if self.run else None

    @property
    def run_url(self) -> Optional[str]:
        """Get current run URL"""
        return self.run.url if self.run else None

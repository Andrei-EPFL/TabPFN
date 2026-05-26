#  Copyright (c) Prior Labs GmbH 2026.

"""Multi-series fine-tuning for FinetunedTabPFNRegressor.

Adds a ``fit_multi`` entry-point that accepts a *list* of time-series tables
(each a pair of X / y arrays).  For every optimiser step the loss is computed
independently for every series in the mini-batch and the *sum* of those losses
is back-propagated in a single ``backward()`` call.

Design principles
-----------------
* **No upstream edits required.**  ``MultiSeriesFinetunedTabPFNRegressor``
  inherits from ``FinetunedTabPFNRegressor`` and overrides only the three
  methods that need changing.
* **Validation** is computed as the mean MSE across all series (each series
  uses its own held-out split).
* The regular ``fit(X, y)`` API still works unchanged — pass a single series
  and it falls back to the parent behaviour.
* ``get_preprocessed_dataset_chunks`` already accepts ``list[XType]`` /
  ``list[YType]``, so the DataLoader naturally yields batches that come from
  *any* of the series.  The per-series loss accumulation is layered on top via
  a **grouped DataLoader** strategy described below.

How the per-series accumulation works
--------------------------------------
Instead of one flat DataLoader over all series we create **one DataLoader per
series** each epoch.  We then zip-iterate them: for each position we draw one
batch from every series, compute the loss, sum, and backprop.

If the DataLoaders have different lengths (series of different sizes) we cycle
the shorter ones so every optimiser step always sees exactly ``n_series``
forward passes.

                ┌──────────────────────────────────────┐
  epoch loop    │  series_1  series_2  …  series_N     │
                │  batch_1   batch_1      batch_1  ──► total_loss = Σ loss_i
                │  batch_2   batch_2      batch_2  ──► total_loss = Σ loss_i
                │  …                                   │
                └──────────────────────────────────────┘
"""

from __future__ import annotations

import copy
import logging
import time
import warnings
from functools import partial
from itertools import cycle
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from tqdm.auto import tqdm

from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.finetuning._torch_compat import GradScaler, autocast, sdpa_kernel_context
from tabpfn.finetuning.data_util import (
    ClassifierBatch,
    RegressorBatch,
    get_preprocessed_dataset_chunks,
    meta_dataset_collator,
)
from tabpfn.finetuning.logging import FinetuningLogger, NullLogger 

from tabpfn.finetuning import FinetunedTabPFNRegressor

from tabpfn.finetuning.train_util import (
    get_and_init_optimizer,
    get_checkpoint_path_and_epoch_from_output_dir,
    get_cosine_schedule_with_warmup,
    save_checkpoint,
)
from tabpfn.utils import infer_devices, infer_random_state
from tabpfn.validation import ensure_compatible_fit_inputs_sklearn

logger = logging.getLogger(__name__)

# Re-export MAX_VALIDATION_SAMPLES from base
MAX_VALIDATION_SAMPLES = 50_000


class MultiSeriesFinetunedTabPFNRegressor(FinetunedTabPFNRegressor):
    """Fine-tune TabPFNRegressor on a *list* of time-series tables.

    New / changed public API
    -------------------------
    ``fit_multi(X_list, y_list, ...)``
        Fine-tune on multiple series at once.  Losses are accumulated across
        all series before each gradient update.

    ``fit(X, y, ...)``
        Works exactly as before (single series, falls back to parent).

    ``predict(X)``
        Unchanged — uses the finetuned inference regressor.

    Parameters (new, multi-series specific)

    """

    def __init__(
        self,
        *,
        n_draw_accum: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_draw_accum = n_draw_accum
    # ------------------------------------------------------------------
    # Public multi-series entry-point
    # ------------------------------------------------------------------

    def fit_multi(
        self,
        X_list: list[XType],
        y_list: list[YType],
        output_dir: Path | None = None
    ) -> "MultiSeriesFinetunedTabPFNRegressor":
        """Fine-tune on multiple time-series tables.

        Args:
            X_list: List of feature matrices, one per time series.
                Each element has shape ``(n_samples_i, n_features)``.
            y_list: List of target vectors, one per time series.
                Each element has shape ``(n_samples_i,)``.
            X_val_list: Optional list of validation feature matrices.
                If provided must match ``X_list`` in length.
            y_val_list: Optional list of validation target vectors.
                If provided must match ``y_list`` in length.
            output_dir: Directory for checkpoint saving.

        Returns:
            The fitted instance.
        """
        if len(X_list) != len(y_list):
            raise ValueError(
                f"X_list and y_list must have the same length, "
                f"got {len(X_list)} and {len(y_list)}."
            )

        if output_dir is None:
            warnings.warn(
                "`output_dir` is not set.  No checkpointing will be done.",
                UserWarning,
                stacklevel=2,
            )
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        return self._fit_multi(
            X_list=X_list,
            y_list=y_list,
            output_dir=output_dir,
        )

    # ------------------------------------------------------------------
    # Internal multi-series training loop
    # ------------------------------------------------------------------

    def _fit_multi(  # noqa: C901, PLR0912, PLR0915
        self,
        X_list: list[XType],
        y_list: list[YType],
        output_dir: Path | None,
    ) -> "MultiSeriesFinetunedTabPFNRegressor":
        """Core multi-series fine-tuning loop."""
        _logger = self.experiment_logger or NullLogger()
        global_step = 0

        config = {
            k: v for k, v in self.get_params().items() if k != "experiment_logger"
        }
        try:
            _logger.setup(config)
        except (OSError, ModuleNotFoundError):
            logger.warning(
                "Experiment logger setup failed, falling back to NullLogger.",
                exc_info=True,
            )
            _logger = NullLogger()

        start_time = time.monotonic()

        _estimator_kwargs = copy.deepcopy(self._estimator_kwargs)
        model_path = _estimator_kwargs.pop("model_path", None)
        inference_config = copy.deepcopy(_estimator_kwargs.get("inference_config", {}))
        base_estimator_config: dict[str, Any] = {
            **_estimator_kwargs,
            "ignore_pretraining_limits": True,
            "device": self.device,
            "random_state": self.random_state,
            "inference_config": inference_config,
        }

        # Config used for the finetuning loop.
        finetuning_estimator_config = self._build_estimator_config(
            base_estimator_config,
            self.n_estimators_finetune,
        )
        if model_path is not None:
            finetuning_estimator_config["model_path"] = model_path

        # Configs used for validation-time evaluation and final inference.
        validation_eval_config = self._build_eval_config(
            base_estimator_config,
            self.n_estimators_validation,
        )
        # final_inference_eval_config = self._build_eval_config(
        #     base_estimator_config,
        #     self.n_estimators_final_inference,
        # )

       
        eval_devices = infer_devices(self.device)
        validation_eval_config["device"] = eval_devices
        # final_inference_eval_config["device"] = eval_devices


        # X, y = X_list[0], y_list[0]
        # # Store the original training size for checkpoint naming
        # train_size = X.shape[0]
        # print("train_size", train_size)

        epoch_to_start_from = 0
        checkpoint_path = None
        # if output_dir is not None:
        #     checkpoint_path, epoch_to_start_from = (
        #         get_checkpoint_path_and_epoch_from_output_dir(
        #             output_dir=output_dir,
        #             train_size=train_size,
        #             get_best=False,
        #         )
        #     )
        #     if checkpoint_path is not None:
        #         logger.info(
        #             f"Restarting training from checkpoint {checkpoint_path} at epoch "
        #             f"{epoch_to_start_from}",
        #         )
        #         finetuning_estimator_config["model_path"] = checkpoint_path

        self.finetuned_estimator_ = self._create_estimator(finetuning_estimator_config)
        self._setup_estimator()

       

        self.finetuned_estimator_._initialize_model_variables()
        self.finetuned_estimator_.model_.to(self.device)

        finetuning_performance_options = PerformanceOptions(
            force_recompute_layer=self.use_activation_checkpointing,
            use_chunkwise_inference=False,
        )

        model_for_optimization = self.finetuned_estimator_.model_


        optimizer = get_and_init_optimizer(
            model_parameters=model_for_optimization.parameters(),  # type: ignore
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            checkpoint_path=checkpoint_path,
            device=self.device,
        )

        use_amp = self.device.startswith("cuda") and torch.cuda.is_available()
        scaler = GradScaler() if use_amp else None  # type: ignore

      
        
        # X_validated, y_validated, self.feature_names_in_, self.n_features_in_ = (
        #     ensure_compatible_fit_inputs_sklearn(
        #         X,
        #         y,
        #         estimator=self.finetuned_estimator_,
        #         ensure_y_numeric=self._model_type == "regressor",
        #     )
        # )
        # self.X_ = X
        # self.y_ = y
        # X, y = X_validated, y_validated

        
        # X_train, X_val, y_train, y_val = self._get_train_val_split(X, y)


        # # --- Initial eval (rank 0 only) ---
        # logger.info("--- 🚀 Eval default model ---")
        # eval_result = self._evaluate_model(
        #     validation_eval_config,
        #     X_train,  # pyright: ignore[reportArgumentType]
        #     y_train,  # pyright: ignore[reportArgumentType]
        #     X_val,  # pyright: ignore[reportArgumentType]
        #     y_val,  # pyright: ignore[reportArgumentType]
        # )
        # self._log_epoch_evaluation(-1, eval_result, mean_train_loss=None)
        # best_metric: float = eval_result.primary

        # n_finetune_ctx_plus_query_samples = min(
        #     self.n_finetune_ctx_plus_query_samples,
        #     len(y_train),
        # )
        
        static_seed, rng = infer_random_state(self.random_state)
        preprocessing_random_state = (
            static_seed if self.use_fixed_preprocessing_seed else rng
        )

        logger.info("--- 🚀 Starting Fine-tuning ---")
        patience_counter = 0
        best_model_state: dict[str, torch.Tensor] | None = None

        scheduler: LambdaLR | None = None

        start_time = time.monotonic()

        # finetuning_query_size = self._get_valid_finetuning_query_size(
        #     query_size=int(
        #         n_finetune_ctx_plus_query_samples * self.finetune_ctx_query_split_ratio
        #     ),
        #     y_train=y_train,
        # )

        # print("n_finetune", n_finetune_ctx_plus_query_samples)
        print("self.finetune", self.finetune_ctx_query_split_ratio)
        # print("finetuning_query_size", finetuning_query_size)
        for epoch in range(epoch_to_start_from, self.epochs):
            # Per-epoch aggregates for cleaner learning curves.
            epoch_loss_sum = 0.0

            epoch_random_state = static_seed + epoch
            optimizer.zero_grad()
            global_batch = 0
            accumulated_loss = torch.tensor(0.0, device=self.device)
            
            for X_i, y_i in zip(X_list, y_list):

                X_validated, y_validated, self.feature_names_in_, self.n_features_in_ = (
                    ensure_compatible_fit_inputs_sklearn(
                        X_i,
                        y_i,
                        estimator=self.finetuned_estimator_,
                        ensure_y_numeric=self._model_type == "regressor",
                    )
                )
                self.X_ = X_i
                self.y_ = y_i
                X, y = X_validated, y_validated
                print("general X and y shapes", X.shape, y.shape)

               
                X_train, X_val, y_train, y_val = self._get_train_val_split(X, y)

                print("train:", X_train.shape, "validation", X_val.shape)

                # Calculate the context size used during finetuning.
                print("self.n_finetune", self.n_finetune_ctx_plus_query_samples)

                n_finetune_ctx_plus_query_samples = min(
                    self.n_finetune_ctx_plus_query_samples,
                    len(y_train),
                )

                finetuning_query_size = self._get_valid_finetuning_query_size(
                   query_size=int(
                        n_finetune_ctx_plus_query_samples * self.finetune_ctx_query_split_ratio
                    ),
                    y_train=y_train,
                )
            
                # Regenerate datasets each epoch with a different random_state
                training_splitter = partial(
                    train_test_split,
                    test_size=finetuning_query_size,
                    random_state=epoch_random_state,
                )

                training_datasets = get_preprocessed_dataset_chunks(
                    calling_instance=self.finetuned_estimator_,
                    X_raw=X_train,
                    y_raw=y_train,
                    split_fn=training_splitter,
                    max_data_size=n_finetune_ctx_plus_query_samples,
                    model_type=self._model_type,
                    equal_split_size=False,
                    data_shuffle_seed=epoch_random_state,
                    preprocessing_random_state=preprocessing_random_state,
                )


                dataloader_generator = torch.Generator().manual_seed(epoch_random_state)
                finetuning_dataloader = DataLoader(
                    training_datasets,
                    batch_size=self.meta_batch_size,
                    collate_fn=meta_dataset_collator,
                    shuffle=True,
                    generator=dataloader_generator,
                )

                # Instantiate the LR scheduler only once
                if self.use_lr_scheduler and scheduler is None:
                    steps_per_epoch = len(finetuning_dataloader)
                    if steps_per_epoch == 0:
                        logger.warning(
                            "No training batches available; ending training early.",
                        )
                        break

                    total_steps = steps_per_epoch * self.epochs
                    warmup_steps = int(total_steps * 0.1)

                    lrate_schedule_fn = get_cosine_schedule_with_warmup(
                        total_steps=total_steps,
                        warmup_steps=warmup_steps,
                        warmup_only=self.lr_warmup_only,
                    )
                    scheduler = LambdaLR(optimizer, lr_lambda=lrate_schedule_fn)

                    logger.info(
                        "Using LambdaLR %s schedule: total_steps=%d, warmup_steps=%d",
                        "warmup-only (constant LR after warmup)"
                        if self.lr_warmup_only
                        else "warmup+cosine",
                        total_steps,
                        warmup_steps,
                    )
                    print("steps_per_epoch:", steps_per_epoch)
                
                progress_bar = tqdm(
                    finetuning_dataloader,
                    desc=f"Finetuning Epoch {epoch + 1}/{self.epochs}"
                )
                for iiii, batch in enumerate(progress_bar):
                    print("batch no", iiii)
                    print("xshape", batch.X_context[0].shape)
                    print("yshape", batch.y_context[0].shape)

                    should_skip = self._should_skip_batch(batch)
                    
                    if should_skip:
                        continue

                    self._setup_batch(batch)

                    self.finetuned_estimator_.fit_from_preprocessed(
                        batch.X_context,
                        batch.y_context,
                        batch.cat_indices,
                        batch.configs,
                        performance_options=finetuning_performance_options,
                    )


                    use_scaler = use_amp and scaler is not None

                    with autocast(enabled=use_scaler), sdpa_kernel_context():  # type: ignore
                        loss = self._forward_with_loss(batch)
                    
                    accumulated_loss = accumulated_loss + loss / self.n_draw_accum # normalize as we go
                    epoch_loss_sum += float(loss.detach().item())
                    
                
                global_batch += 1
                
                is_accum_step = (global_batch % self.n_draw_accum == 0)
                is_last_batch = (global_batch == len(X_list))  # see note below

                if is_accum_step or is_last_batch:

                    if use_scaler:
                        with sdpa_kernel_context():
                            scaler.scale(accumulated_loss).backward()  # type: ignore
                        scaler.unscale_(optimizer)  # type: ignore

                        if self.grad_clip_value is not None:
                            clip_grad_norm_(
                                model_for_optimization.parameters(),
                                self.grad_clip_value,
                            )

                        scaler.step(optimizer)  # type: ignore
                        scaler.update()  # type: ignore
                    else:
                        with sdpa_kernel_context():
                            accumulated_loss.backward()

                        if self.grad_clip_value is not None:
                            clip_grad_norm_(
                                model_for_optimization.parameters(),
                                self.grad_clip_value,
                            )

                        optimizer.step()

                    if scheduler is not None:
                        scheduler.step()


                    global_step += 1

                    current_lr = (
                        scheduler.get_last_lr()[0]
                        if scheduler is not None
                        else self.learning_rate
                    )
                    _logger.log_step(
                        {
                            "train/loss": epoch_loss_sum,
                            "train/lr": current_lr,
                            "train/epoch": epoch,
                            "train/global_step": global_step,
                        },
                        step=global_step,
                    )

                    progress_bar.set_postfix(
                        loss=f"{epoch_loss_sum:.4f}",
                    )

                

            mean_train_loss = (
                epoch_loss_sum
            )

            # --- Validation (rank 0 only), broadcast metric ---
            eval_result = self._evaluate_model(
                validation_eval_config,
                X_train,  # pyright: ignore[reportArgumentType]
                y_train,  # pyright: ignore[reportArgumentType]
                X_val,  # pyright: ignore[reportArgumentType]
                y_val,  # pyright: ignore[reportArgumentType]
            )
            self._log_epoch_evaluation(epoch, eval_result, mean_train_loss)

            epoch_log_metrics: dict[str, float] = {
                "train/epoch": epoch,
                f"val/{self._metric_name}": eval_result.primary,
            }
            if mean_train_loss is not None:
                epoch_log_metrics["train/mean_loss"] = mean_train_loss
            for k, v in eval_result.secondary.items():
                epoch_log_metrics[f"val/{k}"] = v
            _logger.log_epoch(epoch_log_metrics, step=global_step)

            primary_metric = eval_result.primary
            


        #     if (
        #         output_dir is not None
        #         and not np.isnan(primary_metric)
        #     ):
        #         save_interval_checkpoint = (
        #             self.save_checkpoint_interval is not None
        #             and (epoch + 1) % self.save_checkpoint_interval == 0
        #         )

        #         is_best = self._is_improvement(primary_metric, best_metric)

        #         if save_interval_checkpoint or is_best:
        #             save_checkpoint(
        #                 estimator=self.finetuned_estimator_,
        #                 output_dir=output_dir,
        #                 epoch=epoch + 1,
        #                 optimizer=optimizer,
        #                 metrics=self._get_checkpoint_metrics(eval_result),
        #                 train_size=train_size,
        #                 is_best=is_best,
        #                 save_interval_checkpoint=save_interval_checkpoint,
        #             )

        #     if self.early_stopping and not np.isnan(primary_metric):
        #         if self._is_improvement(primary_metric, best_metric):
        #             best_metric = primary_metric
        #             patience_counter = 0
        #             model_sd = self.finetuned_estimator_.model_.state_dict()
        #             best_model_state = {
        #                 k: v.detach().cpu().clone() for k, v in model_sd.items()
        #             }
        #         else:
        #             patience_counter += 1
        #             logger.info(
        #                 "⚠️  No improvement for %s epochs. Best %s: %.4f",
        #                 patience_counter,
        #                 self._metric_name,
        #                 best_metric,
        #             )

        #         if patience_counter >= self.early_stopping_patience:
        #             logger.info(
        #                 "🛑 Early stopping triggered. Best %s: %.4f",
        #                 self._metric_name,
        #                 best_metric,
        #             )
        #             if best_model_state is not None:
        #                 self.finetuned_estimator_.model_.load_state_dict(
        #                     best_model_state
        #                 )
        #             break

        #     if self.time_limit is not None:
        #         elapsed_time = time.monotonic() - start_time
        #         if elapsed_time > self.time_limit:
        #             logger.info(
        #                 "🛑 Time limit of %d seconds reached. Stopping training.",
        #                 self.time_limit,
        #             )
        #             break

        #         n_epochs_run = epoch + 1 - epoch_to_start_from
        #         if elapsed_time + (elapsed_time / n_epochs_run) > self.time_limit:
        #             logger.info(
        #                 "🛑 Not enough time remaining for another epoch. "
        #                 "Stopping training.",
        #             )
        #             break

        # if self.early_stopping and best_model_state is not None:
        #     self.finetuned_estimator_.model_.load_state_dict(best_model_state)

        _logger.finish()
        logger.info("--- ✅ Fine-tuning Finished ---")
        # self._setup_inference_model(final_inference_eval_config)

        self.is_fitted_ = True
        return self


    # ------------------------------------------------------------------
    # Multi-series evaluation helper
    # ------------------------------------------------------------------

    def _evaluate_model_multi(
        self,
        eval_config: dict[str, Any],
        X_trains: list[np.ndarray],
        y_trains: list[np.ndarray],
        X_vals: list[np.ndarray],
        y_vals: list[np.ndarray],
    ) -> EvalResult:
        """Evaluate on all series; return mean MSE as primary metric.

        A separate ``TabPFNRegressor`` clone is fitted on *each* series and
        scored on its corresponding validation split.  The reported MSE is the
        mean over all series (macro average).
        """
        from tabpfn import TabPFNRegressor

        per_series_mse: list[float] = []

        for X_tr, y_tr, X_vl, y_vl in zip(X_trains, y_trains, X_vals, y_vals):
            eval_reg = clone_model_for_evaluation(
                self.finetuned_estimator_,
                eval_config,
                TabPFNRegressor,
            )
            eval_reg.fit(X_tr, y_tr)
            try:
                preds = eval_reg.predict(X_vl)
                mse = mean_squared_error(y_vl, preds)
            except (ValueError, RuntimeError, AttributeError) as exc:
                logger.warning("Evaluation failed for a series: %s", exc)
                mse = np.nan
            per_series_mse.append(float(mse))

        valid_mses = [m for m in per_series_mse if not np.isnan(m)]
        mean_mse = float(np.mean(valid_mses)) if valid_mses else np.nan

        secondary = {f"series_{i}_mse": v for i, v in enumerate(per_series_mse)}
        return EvalResult(primary=mean_mse, secondary=secondary)

    # ------------------------------------------------------------------
    # Final inference model — fit on concatenated data by default
    # ------------------------------------------------------------------

    def _setup_inference_model_multi(
        self, final_inference_eval_config: dict[str, Any]
    ) -> None:
        """Fit the inference model on all series data concatenated."""
        from tabpfn import TabPFNRegressor

        X_all = np.concatenate(self._X_list_, axis=0)  # type: ignore[arg-type]
        y_all = np.concatenate(self._y_list_, axis=0)  # type: ignore[arg-type]

        finetuned_inference_regressor = clone_model_for_evaluation(
            self.finetuned_estimator_,
            final_inference_eval_config,
            TabPFNRegressor,
        )
        self.finetuned_inference_regressor_ = finetuned_inference_regressor
        self.finetuned_inference_regressor_.fit_mode = "fit_preprocessors"  # type: ignore
        self.finetuned_inference_regressor_.fit(X_all, y_all)

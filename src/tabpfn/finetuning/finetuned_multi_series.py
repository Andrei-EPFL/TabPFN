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

from tabpfn.constants import XType, YType
from tabpfn.finetuning.data_util import (
    get_preprocessed_dataset_chunks,
    meta_dataset_collator,
)
from tabpfn.finetuning.finetuned_base import EvalResult
from tabpfn.finetuning.train_util import clone_model_for_evaluation, save_checkpoint
from tabpfn.utils import infer_random_state
from tabpfn.validation import ensure_compatible_fit_inputs_sklearn

# Import the concrete regressor we're extending
from finetuned_regressor import FinetunedTabPFNRegressor  # noqa: E402

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
    ----------------------------------------
    series_loss_reduction : {"sum", "mean"}
        How to combine per-series losses into the scalar that is
        backpropagated.  ``"sum"`` (default) scales the gradient by the
        number of series; ``"mean"`` normalises it.
    """

    def __init__(
        self,
        *,
        series_loss_reduction: Literal["sum", "mean"] = "sum",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.series_loss_reduction = series_loss_reduction

    # ------------------------------------------------------------------
    # Public multi-series entry-point
    # ------------------------------------------------------------------

    def fit_multi(
        self,
        X_list: list[XType],
        y_list: list[YType],
        X_val_list: list[XType] | None = None,
        y_val_list: list[YType] | None = None,
        output_dir: Path | None = None,
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
        if (X_val_list is None) != (y_val_list is None):
            raise ValueError(
                "X_val_list and y_val_list must both be provided or both be None."
            )
        if X_val_list is not None and len(X_val_list) != len(X_list):
            raise ValueError(
                "X_val_list must have the same length as X_list, "
                f"got {len(X_val_list)} and {len(X_list)}."
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
            X_val_list=X_val_list,
            y_val_list=y_val_list,
            output_dir=output_dir,
        )

    # ------------------------------------------------------------------
    # Internal multi-series training loop
    # ------------------------------------------------------------------

    def _fit_multi(  # noqa: C901, PLR0912, PLR0915
        self,
        X_list: list[XType],
        y_list: list[YType],
        X_val_list: list[XType] | None,
        y_val_list: list[YType] | None,
        output_dir: Path | None,
    ) -> "MultiSeriesFinetunedTabPFNRegressor":
        """Core multi-series fine-tuning loop."""
        from tabpfn.architectures.interface import PerformanceOptions
        from tabpfn.finetuning._torch_compat import GradScaler, autocast, sdpa_kernel_context
        from tabpfn.finetuning.train_util import (
            get_and_init_optimizer,
            get_checkpoint_path_and_epoch_from_output_dir,
            get_cosine_schedule_with_warmup,
        )
        from tabpfn.finetuning.logging import NullLogger
        from torch.optim.lr_scheduler import LambdaLR
        from tqdm.auto import tqdm

        _logger = self.experiment_logger or NullLogger()
        start_time = time.monotonic()

        # ── Validate & convert every series ────────────────────────────
        if self.eval_metric is None:
            self.eval_metric = "mse"

        # Build a temporary estimator just for input validation
        _estimator_kwargs = copy.deepcopy(self._estimator_kwargs)
        model_path = _estimator_kwargs.pop("model_path", None)
        base_estimator_config: dict[str, Any] = {
            **_estimator_kwargs,
            "ignore_pretraining_limits": True,
            "device": self.device,
            "random_state": self.random_state,
        }
        finetuning_estimator_config = self._build_estimator_config(
            base_estimator_config, self.n_estimators_finetune
        )
        if model_path is not None:
            finetuning_estimator_config["model_path"] = model_path

        # Check/resume from checkpoint
        train_size_total = sum(len(y) for y in y_list)
        epoch_to_start_from = 0
        checkpoint_path = None
        if output_dir is not None:
            checkpoint_path, epoch_to_start_from = (
                get_checkpoint_path_and_epoch_from_output_dir(
                    output_dir=output_dir,
                    train_size=train_size_total,
                    get_best=False,
                )
            )
            if checkpoint_path is not None:
                logger.info(
                    "Restarting from checkpoint %s at epoch %d",
                    checkpoint_path,
                    epoch_to_start_from,
                )
                finetuning_estimator_config["model_path"] = checkpoint_path

        self.finetuned_estimator_ = self._create_estimator(finetuning_estimator_config)
        self._setup_estimator()

        # Validate inputs and split train/val for every series
        X_trains: list[np.ndarray] = []
        y_trains: list[np.ndarray] = []
        X_vals: list[np.ndarray] = []
        y_vals: list[np.ndarray] = []

        # Store originals for final inference
        self.X_ = X_list[0]   # kept for _setup_inference_model compat; overridden below
        self.y_ = y_list[0]
        self._X_list_ = X_list
        self._y_list_ = y_list

        for i, (X_i, y_i) in enumerate(zip(X_list, y_list)):
            X_v, y_v, fn, nf = ensure_compatible_fit_inputs_sklearn(
                X_i, y_i,
                estimator=self.finetuned_estimator_,
                ensure_y_numeric=True,
            )
            if i == 0:
                self.feature_names_in_ = fn
                self.n_features_in_ = nf

            if X_val_list is not None:
                X_tr, y_tr = X_v, y_v
                X_vl, y_vl, _, _ = ensure_compatible_fit_inputs_sklearn(
                    X_val_list[i], y_val_list[i],  # type: ignore[index]
                    estimator=self.finetuned_estimator_,
                    ensure_y_numeric=True,
                )
            else:
                X_tr, X_vl, y_tr, y_vl = self._get_train_val_split(X_v, y_v)

            X_trains.append(X_tr)
            y_trains.append(y_tr)
            X_vals.append(X_vl)
            y_vals.append(y_vl)

        n_series = len(X_trains)

        # ── Model + optimiser setup ─────────────────────────────────────
        self.finetuned_estimator_._initialize_model_variables()
        self.finetuned_estimator_.model_.to(self.device)

        finetuning_performance_options = PerformanceOptions(
            force_recompute_layer=self.use_activation_checkpointing,
            use_chunkwise_inference=False,
        )

        optimizer = get_and_init_optimizer(
            model_parameters=self.finetuned_estimator_.model_.parameters(),
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            checkpoint_path=checkpoint_path,
            device=self.device,
        )

        use_amp = self.device.startswith("cuda") and torch.cuda.is_available()
        scaler = GradScaler() if use_amp else None  # type: ignore

        # ── Validation config ───────────────────────────────────────────
        validation_eval_config = self._build_eval_config(
            base_estimator_config, self.n_estimators_validation
        )
        final_inference_eval_config = self._build_eval_config(
            base_estimator_config, self.n_estimators_final_inference
        )
        from tabpfn.utils import infer_devices
        eval_devices = infer_devices(self.device)
        validation_eval_config["device"] = eval_devices
        final_inference_eval_config["device"] = eval_devices

        # ── Initial evaluation ──────────────────────────────────────────
        logger.info("--- 🚀 Eval default model (multi-series) ---")
        eval_result = self._evaluate_model_multi(
            validation_eval_config, X_trains, y_trains, X_vals, y_vals
        )
        self._log_epoch_evaluation(-1, eval_result, mean_train_loss=None)
        best_metric: float = eval_result.primary
        best_model_state: dict[str, torch.Tensor] | None = None
        patience_counter = 0

        static_seed, rng = infer_random_state(self.random_state)
        preprocessing_random_state = (
            static_seed if self.use_fixed_preprocessing_seed else rng
        )

        finetuning_query_sizes = [
            self._get_valid_finetuning_query_size(
                query_size=int(
                    min(self.n_finetune_ctx_plus_query_samples, len(y_tr))
                    * self.finetune_ctx_query_split_ratio
                ),
                y_train=y_tr,
            )
            for y_tr in y_trains
        ]

        scheduler: LambdaLR | None = None
        global_step = 0

        logger.info("--- 🚀 Starting Multi-Series Fine-tuning (%d series) ---", n_series)

        for epoch in range(epoch_to_start_from, self.epochs):
            epoch_loss_sum = 0.0
            epoch_batches = 0
            epoch_random_state = static_seed + epoch

            # ── Build one DataLoader per series ─────────────────────────
            training_splitter_base = partial(train_test_split, random_state=epoch_random_state)

            series_dataloaders: list[DataLoader] = []
            for s_idx, (X_tr, y_tr, q_size) in enumerate(
                zip(X_trains, y_trains, finetuning_query_sizes)
            ):
                n_ctx_plus_q = min(self.n_finetune_ctx_plus_query_samples, len(y_tr))
                splitter = partial(training_splitter_base, test_size=q_size)

                ds = get_preprocessed_dataset_chunks(
                    calling_instance=self.finetuned_estimator_,
                    X_raw=X_tr,
                    y_raw=y_tr,
                    split_fn=splitter,
                    max_data_size=n_ctx_plus_q,
                    model_type=self._model_type,
                    equal_split_size=False,
                    data_shuffle_seed=epoch_random_state + s_idx,
                    preprocessing_random_state=preprocessing_random_state,
                )

                dl_gen = torch.Generator().manual_seed(epoch_random_state + s_idx)
                series_dataloaders.append(
                    DataLoader(
                        ds,
                        batch_size=self.meta_batch_size,
                        collate_fn=meta_dataset_collator,
                        shuffle=True,
                        generator=dl_gen,
                    )
                )

            # ── LR scheduler (initialised once) ─────────────────────────
            if self.use_lr_scheduler and scheduler is None:
                max_steps_per_epoch = max(len(dl) for dl in series_dataloaders)
                if max_steps_per_epoch == 0:
                    logger.warning("No training batches; stopping early.")
                    break
                total_steps = max_steps_per_epoch * self.epochs
                warmup_steps = int(total_steps * 0.1)
                lrate_fn = get_cosine_schedule_with_warmup(
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    warmup_only=self.lr_warmup_only,
                )
                scheduler = LambdaLR(optimizer, lr_lambda=lrate_fn)
                logger.info(
                    "LR schedule: total_steps=%d, warmup_steps=%d",
                    total_steps, warmup_steps,
                )

            # ── Zip-iterate series DataLoaders ───────────────────────────
            # Use the longest DataLoader as the "clock"; cycle shorter ones.
            max_len = max(len(dl) for dl in series_dataloaders)
            cycled_iters = [
                cycle(dl) if len(dl) < max_len else iter(dl)
                for dl in series_dataloaders
            ]

            pbar = tqdm(
                range(max_len),
                desc=f"Finetuning Epoch {epoch + 1}/{self.epochs}",
            )

            for _ in pbar:
                optimizer.zero_grad()

                total_loss = torch.tensor(0.0, device=self.device)
                n_valid = 0

                # ── Per-series forward pass ──────────────────────────────
                for s_iter in cycled_iters:
                    batch = next(s_iter)

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

                    with autocast(enabled=(use_amp and scaler is not None)), sdpa_kernel_context():  # type: ignore
                        loss_i = self._forward_with_loss(batch)

                    total_loss = total_loss + loss_i
                    n_valid += 1

                if n_valid == 0:
                    continue  # all series skipped this step

                # ── Reduce across series ─────────────────────────────────
                if self.series_loss_reduction == "mean" and n_valid > 0:
                    total_loss = total_loss / n_valid

                # ── Backward + optimiser step ────────────────────────────
                use_scaler = use_amp and scaler is not None
                if use_scaler:
                    from tabpfn.finetuning._torch_compat import sdpa_kernel_context as _sdpa
                    with _sdpa():
                        scaler.scale(total_loss).backward()  # type: ignore
                    scaler.unscale_(optimizer)  # type: ignore
                    if self.grad_clip_value is not None:
                        clip_grad_norm_(
                            self.finetuned_estimator_.model_.parameters(),
                            self.grad_clip_value,
                        )
                    scaler.step(optimizer)  # type: ignore
                    scaler.update()  # type: ignore
                else:
                    from tabpfn.finetuning._torch_compat import sdpa_kernel_context as _sdpa
                    with _sdpa():
                        total_loss.backward()
                    if self.grad_clip_value is not None:
                        clip_grad_norm_(
                            self.finetuned_estimator_.model_.parameters(),
                            self.grad_clip_value,
                        )
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                loss_scalar = float(total_loss.detach().item())
                epoch_loss_sum += loss_scalar
                epoch_batches += 1
                global_step += 1
                pbar.set_postfix(total_loss=f"{loss_scalar:.4f}", n_series=n_valid)

            mean_train_loss = epoch_loss_sum / epoch_batches if epoch_batches > 0 else None

            # ── Validation ───────────────────────────────────────────────
            eval_result = self._evaluate_model_multi(
                validation_eval_config, X_trains, y_trains, X_vals, y_vals
            )
            self._log_epoch_evaluation(epoch, eval_result, mean_train_loss)
            primary_metric = eval_result.primary

            # ── Checkpoint ──────────────────────────────────────────────
            if output_dir is not None and not np.isnan(primary_metric):
                save_interval = (
                    self.save_checkpoint_interval is not None
                    and (epoch + 1) % self.save_checkpoint_interval == 0
                )
                is_best = self._is_improvement(primary_metric, best_metric)
                if save_interval or is_best:
                    save_checkpoint(
                        estimator=self.finetuned_estimator_,
                        output_dir=output_dir,
                        epoch=epoch + 1,
                        optimizer=optimizer,
                        metrics=self._get_checkpoint_metrics(eval_result),
                        train_size=train_size_total,
                        is_best=is_best,
                        save_interval_checkpoint=save_interval,
                    )

            # ── Early stopping ───────────────────────────────────────────
            if self.early_stopping and not np.isnan(primary_metric):
                if self._is_improvement(primary_metric, best_metric):
                    best_metric = primary_metric
                    patience_counter = 0
                    sd = self.finetuned_estimator_.model_.state_dict()
                    best_model_state = {
                        k: v.detach().cpu().clone() for k, v in sd.items()
                    }
                else:
                    patience_counter += 1
                    logger.info(
                        "⚠️  No improvement for %d epochs. Best MSE: %.4f",
                        patience_counter, best_metric,
                    )
                if patience_counter >= self.early_stopping_patience:
                    logger.info("🛑 Early stopping. Best MSE: %.4f", best_metric)
                    if best_model_state is not None:
                        self.finetuned_estimator_.model_.load_state_dict(best_model_state)
                    break

            # ── Time limit ───────────────────────────────────────────────
            if self.time_limit is not None:
                elapsed = time.monotonic() - start_time
                n_done = epoch + 1 - epoch_to_start_from
                if elapsed > self.time_limit or elapsed + elapsed / n_done > self.time_limit:
                    logger.info("🛑 Time limit reached.")
                    break

        if self.early_stopping and best_model_state is not None:
            self.finetuned_estimator_.model_.load_state_dict(best_model_state)

        logger.info("--- ✅ Multi-Series Fine-tuning Finished ---")

        # ── Final inference model ────────────────────────────────────────
        # Use first series as canonical fit for inference; you can override.
        self._setup_inference_model_multi(final_inference_eval_config)

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

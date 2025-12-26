import os
import time
import uuid
import string
import random
import asyncio
import json
import threading
import glob
from datetime import datetime
from typing import List
from pydantic import BaseModel, Field
from fastapi import Body, HTTPException, Header, Query  # noqa NID002
from async_timeout import timeout
import re
import shutil
from fastapi.responses import StreamingResponse  # noqa NID002
import requests
from urllib.parse import unquote

import lazyllm
from lazyllm.launcher import Status
from lazyllm.module.llms.utils import uniform_sft_dataset
from lazyllm import FastapiApp as app, LOG as logger
from lazyllm.tools.services import ServerBase

DEFAULT_TOKEN = 'default_token'
_DEFAULT_BODY_EMBED = Body(embed=True)
_DEFAULT_BODY_OPTIONAL = Body(None, embed=True)
_DEFAULT_HEADER_TOKEN = Header(DEFAULT_TOKEN)

def is_url(path):
    return bool(re.match(r'^https?://', path))

def get_filename_from_url(url: str, timeout: int = 10) -> str:
    '''
    Get filename from URL Content-Disposition header
    '''
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()

        cd = resp.headers.get('Content-Disposition', '')
        for pattern in [r'filename\*=UTF-8\'\'(.+)', r'filename=\'?([^\']+)\'?']:
            match = re.search(pattern, cd)
            if match:
                return unquote(match.group(1))
    except Exception:
        pass

    raise HTTPException(status_code=404, detail='File format is not clear')


class Dataset(BaseModel):
    dataset_download_uri: str
    format: int
    dataset_id: str

class TrainingArgs(BaseModel):
    val_size: float = 0.02
    num_train_epochs: int = 1
    learning_rate: float = 0.1
    lr_scheduler_type: str = 'cosine'
    per_device_train_batch_size: int = 32
    cutoff_len: int = 1024
    finetuning_type: str = 'lora'
    lora_rank: int = 8
    lora_alpha: int = 32
    trust_remote_code: bool = True
    ngpus: int = 1

    class Config:
        extra = 'allow'  # extra fields are allowed

class _JobDescription(BaseModel):
    name: str
    model: str
    training_args: TrainingArgs = Field(default_factory=TrainingArgs)
    training_dataset: List[Dataset] = []
    validation_dataset: List[Dataset] = []
    validate_dataset_split_percent: float = Field(default=0.0)
    stage: str = ''

class ModelExport(BaseModel):
    name: str
    model_display_name: str
    model_id: str


class TrainServer(ServerBase):
    # Compiled regex patterns for performance optimization
    _re_progress_bar = re.compile(r'(\d+)%\s*\|\s*[█▉▊▋▌▍▎▏\s]+\s*\|\s*(\d+)/(\d+)')
    _re_non_training_patterns = [
        re.compile(r'Loading checkpoint shards', re.IGNORECASE),
        re.compile(r'Converting format of dataset', re.IGNORECASE),
        re.compile(r'Running tokenizer on dataset', re.IGNORECASE),
    ]
    _re_time_format = re.compile(r'\[\d+:\d+<\d+:\d+,\s*[\d.]+\s*(?:s/it|it/s)\]')
    _re_loss = re.compile(r"'loss':\s*([\d.]+)")
    _re_epoch = re.compile(r"'epoch':\s*([\d.]+)")
    _re_metrics = re.compile(r"\{'loss':\s*([\d.]+).*'epoch':\s*([\d.]+)")
    _re_total_steps = re.compile(r'Total optimization steps\s*=\s*(\d+)')
    _re_step_ratio = re.compile(r'(\d+)/(\d+)')

    def _calculate_final_cost(self, info):
        '''
        Calculate final cost for a job when it's completed (Failed or Done).
        '''
        previous_cost = info.get('cost', 0) or 0
        if info.get('last_cost_update_time') and info.get('started_at'):
            # Use incremental calculation to avoid double counting
            last_update_time = datetime.strptime(info['last_cost_update_time'], self._time_format)
            current_segment_cost = (datetime.now() - last_update_time).total_seconds()
            total_cost = previous_cost + current_segment_cost
        elif info.get('started_at'):
            # If no last_cost_update_time, use started_at for calculation (backward compatible)
            started_at = datetime.strptime(info['started_at'], self._time_format)
            total_cost = (datetime.now() - started_at).total_seconds()
        else:
            # If no started_at, use existing cost
            total_cost = previous_cost
        return total_cost

    def _verify_final_model_files(self, fine_tuned_model_path, finetuning_type='lora'):
        '''
        Verify that final model files are complete and valid.
        Returns True if model files are complete, False otherwise.
        '''
        if not fine_tuned_model_path or not os.path.exists(fine_tuned_model_path):
            return False

        try:
            # Check if merge_path exists (merged model is preferred)
            if 'lazyllm_lora' in fine_tuned_model_path:
                merge_path = fine_tuned_model_path.replace('lazyllm_lora', 'lazyllm_merge')
                if os.path.exists(merge_path):
                    fine_tuned_model_path = merge_path

            if finetuning_type == 'full':
                # For full finetuning, check for config.json and model files
                config_path = os.path.join(fine_tuned_model_path, 'config.json')
                if not os.path.exists(config_path):
                    return False

                # Check for model files (safetensors, bin, or index.json for sharded models)
                model_files = [
                    'model.safetensors',
                    'pytorch_model.bin',
                    'model.safetensors.index.json'
                ]
                has_model = any(
                    os.path.exists(os.path.join(fine_tuned_model_path, f))
                    for f in model_files
                )
                return has_model
            else:
                # For LoRA/QLoRA, check for adapter files
                adapter_config = os.path.join(fine_tuned_model_path, 'adapter_config.json')
                if not os.path.exists(adapter_config):
                    return False

                # Check for adapter weight files
                adapter_files = [
                    'adapter_model.safetensors',
                    'adapter_model.bin'
                ]
                has_adapter = any(
                    os.path.exists(os.path.join(fine_tuned_model_path, f))
                    for f in adapter_files
                )
                return has_adapter
        except Exception:
            return False

    def _check_log_for_errors(self, log_path, job_id, info=None):
        # Priority 1: Verify model files first (most reliable indicator)
        if info:
            fine_tuned_model = info.get('fine_tuned_model')
            if fine_tuned_model:
                finetuning_type = info.get('hyperparameters', {}).get('finetuning_type', 'lora')
                if self._verify_final_model_files(fine_tuned_model, finetuning_type):
                    # Model files are complete, training succeeded regardless of log content
                    logger.info(
                        f'[_check_log_for_errors] Job {job_id} model files are complete, '
                        f'training succeeded')
                    return False

        # Priority 2: If model files are incomplete, use original log check logic (conservative approach)
        if not log_path or not os.path.exists(log_path):
            return False

        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                log_content = f.read()
                # Check for common failure indicators (original logic - conservative)
                error_indicators = [
                    # Memory errors
                    'OutOfMemoryError', 'CUDA out of memory', 'torch.OutOfMemoryError',
                    # Process/Subprocess errors
                    'CalledProcessError', 'returned non-zero exit status',
                    'ChildFailedError', 'FAILED',
                    # File system errors
                    'FileNotFoundError', 'OSError', 'PermissionError',
                    'No space left', 'Disk quota exceeded',
                    # Python standard exceptions
                    'RuntimeError', 'ValueError', 'TypeError',
                    'KeyError', 'AttributeError', 'ImportError',
                    'AssertionError', 'IndexError',
                    # CUDA/NCCL errors
                    'CUDA error', 'NCCL error', 'cuda runtime error',
                    'NCCL initialization', 'NCCL communicator',
                    # Training framework errors
                    'Training failed', 'Trainer crashed', 'Training crashed',
                    # System/Connection errors
                    'Broken pipe', 'Connection refused', 'Connection reset',
                    'TimeoutError', 'ConnectionError',
                    # Traceback indicates serious errors
                    'Traceback (most recent call last)'
                ]
                if any(error in log_content for error in error_indicators):
                    logger.info(
                        f'[_check_log_for_errors] Job {job_id} marked as Done but found '
                        f'errors in log, changing to Failed')
                    return True
        except Exception:
            pass

        return False

    def _update_status(self, token, job_id):
        if not self._in_active_jobs(token, job_id):
            return

        info = self._read_user_job_info(token, job_id)
        # Suspended status is manually paused, should not be auto-updated
        if info.get('status') == 'Suspended':
            return

        try:
            m, _ = self._read_active_job(token, job_id)
            status = m.status(info['model_id']).name
        except Exception:
            try:
                _, thread = self._read_active_job(token, job_id)
                if thread and not thread.is_alive():
                    logger.info(
                        f'[_update_status] Job {job_id} thread is dead, '
                        f'marking as Failed')
                    update = {'status': 'Failed'}
                    self._update_user_job_info(token, job_id, update)
                    try:
                        m, _ = self._pop_active_job(token, job_id)
                    except Exception:
                        pass
                    return
            except Exception:
                pass
            return

        update = {'status': status}

        # Some tasks cannot obtain the storage path when they are just started
        if not info.get('fine_tuned_model'):
            try:
                update['fine_tuned_model'] = self._get_save_path(m)
            except Exception:
                pass
        if not info.get('log_path'):
            try:
                update['log_path'] = m.log_path(info['model_id'])
            except Exception:
                pass

        if Status[status] == Status.Running:
            if not info.get('started_at'):
                update['started_at'] = datetime.now().strftime(self._time_format)

            now = datetime.now()
            started_at_str = update.get('started_at') or info.get('started_at')
            started_at = datetime.strptime(started_at_str, self._time_format)

            last_update_time_str = info.get('last_cost_update_time')
            if last_update_time_str:
                last_update_time = datetime.strptime(last_update_time_str, self._time_format)
                current_segment_cost = (now - last_update_time).total_seconds()
            else:
                current_segment_cost = (now - started_at).total_seconds()

            previous_cost = info.get('cost', 0) or 0
            total_cost = previous_cost + current_segment_cost
            update['cost'] = total_cost
            update['last_cost_update_time'] = now.strftime(self._time_format)

            progress_info = self._extract_training_progress(info.get('log_path'))
            if progress_info:
                # Save progress_percent to user_job_info for display (not stored in database)
                if 'percent' in progress_info:
                    update['progress_percent'] = progress_info['percent']
                # Use total_steps from progress_info if available (avoid duplicate file reading)
                total_steps = progress_info.get('total_steps')
            else:
                total_steps = None

            # If no progress from log, try to initialize from checkpoint step
            if 'progress_percent' not in update:
                checkpoint_step = info.get('checkpoint_step')
                if checkpoint_step is not None and isinstance(checkpoint_step, int):
                    # Use total_steps from progress_info if available (already read from file)
                    if total_steps is None:
                        # Fallback: try to get total steps from log (if not already extracted)
                        log_path = info.get('log_path')
                        if log_path and os.path.exists(log_path):
                            try:
                                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    lines = f.readlines()
                                    # First, try to find 'Total optimization steps' line (most accurate)
                                    # Search from the end to get the latest total steps (after resume)
                                    for line in reversed(lines):
                                        if 'Total optimization steps' in line:
                                            total_steps_match = self._re_total_steps.search(line)
                                            if total_steps_match:
                                                total_steps = int(total_steps_match.group(1))
                                                break

                                    # If not found, try to find from progress bar (less accurate)
                                    if total_steps is None:
                                        for line in reversed(lines[-100:]):
                                            # Skip non-training progress bars
                                            is_non_training = any(
                                                pattern.search(line)
                                                for pattern in self._re_non_training_patterns)
                                            if is_non_training:
                                                continue

                                            # Match progress bar format: ' 50/150 [08:11<00:33,  1.11s/it]'
                                            progress_match = self._re_step_ratio.search(line)
                                            if progress_match:
                                                potential_total = int(progress_match.group(2))
                                                # Only use if reasonable (>= checkpoint_step and <= 10000)
                                                if (potential_total >= checkpoint_step
                                                        and potential_total <= 10000):
                                                    total_steps = potential_total
                                                    break
                            except Exception:
                                pass

                    if total_steps and total_steps > 0:
                        # Use current total steps (after resume) for accurate calculation
                        base_percent = int((checkpoint_step / total_steps) * 100)
                        update['progress_percent'] = max(0, min(100, base_percent))

                # Clear progress_percent if no progress info available
                if 'progress_percent' not in update:
                    if 'progress_percent' in info:
                        update['progress_percent'] = None

        self._update_user_job_info(token, job_id, update)

        if Status[status] == Status.Failed:
            update['progress_percent'] = None
            if 'cost' not in update:
                total_cost = self._calculate_final_cost(info)
                self._update_user_job_info(
                    token, job_id, {'cost': total_cost, 'progress_percent': None})
                logger.info(
                    f'[_update_status] Job {job_id} failed, '
                    f'final cost: {total_cost}s')
            else:
                update['progress_percent'] = None
                logger.info(
                    f'[_update_status] Job {job_id} failed, '
                    f'cost: {update["cost"]}s')

            try:
                m, _ = self._pop_active_job(token, job_id)
                m.stop(info['model_id'])
            except Exception as e:
                logger.info(f'[_update_status] Failed to stop job {job_id} after Failed status: {e}')
            return

        if Status[status] == Status.Done:
            log_path = info.get('log_path') or update.get('log_path')
            # Pass info to allow model file verification
            should_mark_failed = self._check_log_for_errors(log_path, job_id, info)

            if should_mark_failed:
                update['status'] = 'Failed'
                status = 'Failed'
                self._update_user_job_info(token, job_id, update)
                if 'cost' not in update:
                    total_cost = self._calculate_final_cost(info)
                    self._update_user_job_info(
                        token, job_id, {'cost': total_cost, 'progress_percent': None})

                try:
                    m, _ = self._pop_active_job(token, job_id)
                    m.stop(info['model_id'])
                except Exception as e:
                    logger.info(f'[_update_status] Failed to stop job {job_id} after Failed status: {e}')
                return

            update['progress_percent'] = None
            if 'cost' not in update:
                total_cost = self._calculate_final_cost(info)
                self._update_user_job_info(
                    token, job_id, {'cost': total_cost, 'progress_percent': None})
                logger.info(
                    f'[_update_status] Job {job_id} completed, '
                    f'final cost: {total_cost}s')
            else:
                update['progress_percent'] = None

            try:
                m, _ = self._pop_active_job(token, job_id)
            except Exception as e:
                logger.info(f'[_update_status] Failed to pop job {job_id} after Done status: {e}')
            return

        create_time = datetime.strptime(info['created_at'], self._time_format)
        delta_time = (datetime.now() - create_time).total_seconds()

        if delta_time > 300 and Status[status] == Status.Cancelled:
            m, _ = self._pop_active_job(token, job_id)
            m.stop(info['model_id'])
            if info.get('started_at') and not info.get('cost'):
                cost = (datetime.now() - datetime.strptime(info['started_at'], self._time_format)).total_seconds()
                self._update_user_job_info(token, job_id, {'cost': cost})
            return

        if delta_time > 3000 and Status[status] in (Status.TBSubmitted, Status.InQueue, Status.Pending):
            m, _ = self._pop_active_job(token, job_id)
            m.stop(info['model_id'])
            return

    def _get_save_path(self, model):
        if not hasattr(model._impl, '_temp_finetuned_model_path'):
            return None
        return model._impl._temp_finetuned_model_path

    def _extract_training_progress(self, log_path):
        if not log_path or not os.path.exists(log_path):
            return None

        try:
            # Read the last few lines of the log file (progress info is usually at the end)
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

                # Find the LAST 'Running training' or 'Continuing training' line to handle resume
                # This ensures we get progress from the most recent training session
                training_start_idx = None
                last_500_lines = lines[-500:]  # Search in last 500 lines to handle resume cases
                for idx in range(len(last_500_lines) - 1, -1, -1):  # Search backwards
                    line = last_500_lines[idx]
                    if (('Running training' in line
                         or 'Continuing training from checkpoint' in line
                         or 'Total optimization steps' in line)):
                        training_start_idx = idx
                        break

                # Only search in lines after the LAST 'Running training' (if found)
                if training_start_idx is not None:
                    search_lines = last_500_lines[training_start_idx + 1:]
                else:
                    search_lines = lines[-200:]

                # Also search for total_steps from 'Total optimization steps' line
                # This avoids duplicate file reading in _update_status
                total_steps = None
                for line in reversed(lines):
                    if 'Total optimization steps' in line:
                        total_steps_match = self._re_total_steps.search(line)
                        if total_steps_match:
                            total_steps = int(total_steps_match.group(1))
                            break

                # Search backwards for progress information
                for line in reversed(search_lines):
                    original_line = line
                    line = line.strip()
                    if not line:
                        continue

                    # Match progress bar format: ' 75%|███████▌  | 90/120 [08:11<00:33,  1.11s/it]'
                    progress_match = self._re_progress_bar.search(line)
                    if progress_match:
                        # Exclude non-training progress bars with descriptive prefixes
                        is_non_training = any(
                            pattern.search(original_line)
                            for pattern in self._re_non_training_patterns)

                        # Only extract training progress (has time format and
                        # no descriptive prefix)
                        if not is_non_training:
                            # Check if has time format (training progress bar has
                            # [HH:MM<HH:MM, X.XXs/it] or [HH:MM<HH:MM, X.XXit/s])
                            # Support both 's/it' and 'it/s' formats
                            if self._re_time_format.search(original_line):
                                percent = int(progress_match.group(1))
                                current_step = int(progress_match.group(2))
                                progress_total_steps = int(progress_match.group(3))

                                # Use total_steps from 'Total optimization steps' if available,
                                # otherwise use progress_total_steps from progress bar
                                final_total_steps = total_steps if total_steps is not None else progress_total_steps

                                # Try to extract loss information
                                loss_match = self._re_loss.search(line)
                                loss = float(loss_match.group(1)) if loss_match else None

                                # Try to extract epoch information
                                epoch_match = self._re_epoch.search(line)
                                epoch = float(epoch_match.group(1)) if epoch_match else None

                                result = {
                                    'percent': percent,
                                    'step': f'{current_step}/{final_total_steps}',
                                    'loss': loss,
                                    'epoch': epoch
                                }
                                # Add total_steps to result if found, to avoid duplicate file reading
                                if total_steps is not None:
                                    result['total_steps'] = total_steps
                                return result

                    # Match training metrics format: '{'loss': 0.6341, 'epoch': 20.0, ...}'
                    metrics_match = self._re_metrics.search(line)
                    if metrics_match:
                        loss = float(metrics_match.group(1))
                        epoch = float(metrics_match.group(2))
                        result = {
                            'loss': loss,
                            'epoch': epoch
                        }
                        # Add total_steps to result if found, to avoid duplicate file reading
                        if total_steps is not None:
                            result['total_steps'] = total_steps
                        return result

                # If no progress found but total_steps was found, return it
                if total_steps is not None:
                    return {'total_steps': total_steps}
        except Exception as e:
            logger.debug(f'[_extract_training_progress] Failed to extract progress from {log_path}: {e}')

        return None

    def _extract_checkpoint_step(self, checkpoint_path):
        '''Extract step number from checkpoint path (e.g., checkpoint-40 -> 40)'''
        if not checkpoint_path:
            return None
        try:
            checkpoint_name = os.path.basename(checkpoint_path)
            if checkpoint_name.startswith('checkpoint-'):
                step_str = checkpoint_name.split('-')[1]
                if step_str.isdigit():
                    return int(step_str)
        except Exception:
            pass
        return None

    def _is_checkpoint_complete(self, checkpoint_path, finetuning_type='lora'):
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return False

        try:
            if finetuning_type == 'full':
                # For full finetuning, trainer_state.json is required for resume
                trainer_state_path = os.path.join(checkpoint_path, 'trainer_state.json')
                if not os.path.exists(trainer_state_path):
                    return False
                # Also check for model files
                if not os.path.exists(os.path.join(checkpoint_path, 'config.json')) and \
                   not os.path.exists(os.path.join(checkpoint_path, 'model.safetensors.index.json')):
                    return False
                return True
            else:
                # For LoRA/QLoRA, adapter_config.json is required
                if os.path.exists(os.path.join(checkpoint_path, 'adapter_config.json')):
                    return True
                return False
        except Exception:
            return False

    def _get_complete_checkpoint(self, lora_dir, finetuning_type='lora'):
        if not lora_dir or not os.path.exists(lora_dir):
            return None, None

        try:
            checkpoint_dirs = [
                d for d in os.listdir(lora_dir)
                if d.startswith('checkpoint-') and os.path.isdir(os.path.join(lora_dir, d))
            ]
            if not checkpoint_dirs:
                return None, None

            checkpoint_dirs.sort(
                key=lambda d: int(d.split('-')[1]) if d.split('-')[1].isdigit() else 0,
                reverse=True)

            # Check the latest checkpoint first
            latest_path = os.path.join(lora_dir, checkpoint_dirs[0])
            if self._is_checkpoint_complete(latest_path, finetuning_type):
                return latest_path, None

            # Latest checkpoint is incomplete, return the second latest (which must be complete)
            if len(checkpoint_dirs) >= 2:
                second_latest_path = os.path.join(lora_dir, checkpoint_dirs[1])
                return second_latest_path, latest_path

            # Only one checkpoint and it's incomplete
            return None, latest_path
        except Exception as e:
            logger.info(f'[_get_complete_checkpoint] Error: {e}')
            return None, None

    def _get_lora_dir(self, fine_tuned_model):
        if not fine_tuned_model:
            return None
        if 'lazyllm_lora' in fine_tuned_model:
            if fine_tuned_model.endswith('lazyllm_lora'):
                return fine_tuned_model
            else:
                return fine_tuned_model[:fine_tuned_model.find('lazyllm_lora') + len('lazyllm_lora')]
        elif 'lazyllm_merge' in fine_tuned_model:
            return fine_tuned_model.replace('lazyllm_merge', 'lazyllm_lora')
        else:
            return os.path.join(os.path.dirname(fine_tuned_model), 'lazyllm_lora')

    def _start_periodic_checkpoint_cleanup(self, token, job_id):
        '''
        Start a daemon thread to periodically clean up old checkpoints.
        The thread will run every 60 seconds and clean up checkpoints while the job is active.
        '''
        def periodic_cleanup():
            time.sleep(30)  # Initial delay before first cleanup
            while True:
                try:
                    time.sleep(60)  # Run cleanup every 60 seconds
                    if not self._in_active_jobs(token, job_id):
                        break
                    info = self._read_user_job_info(token, job_id)
                    if not info:
                        break
                    lora_dir = self._get_lora_dir(info.get('fine_tuned_model'))
                    if lora_dir and os.path.exists(lora_dir):
                        # Get protected checkpoint path from hyperparameters (if resume is in progress)
                        protected_checkpoint = None
                        hyperparameters = info.get('hyperparameters', {})
                        if hyperparameters and 'resume_from_checkpoint' in hyperparameters:
                            protected_checkpoint = hyperparameters['resume_from_checkpoint']
                        self._cleanup_old_checkpoints(
                            lora_dir, keep_count=3,
                            protected_checkpoint_path=protected_checkpoint)
                except Exception:
                    pass

        cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
        cleanup_thread.start()

    def _cleanup_old_checkpoints(self, lora_dir, keep_count=3, protected_checkpoint_path=None):
        if not lora_dir or not os.path.exists(lora_dir):
            return

        try:
            checkpoint_dirs = [
                d for d in os.listdir(lora_dir)
                if d.startswith('checkpoint-') and os.path.isdir(os.path.join(lora_dir, d))
            ]
            if len(checkpoint_dirs) <= keep_count:
                logger.debug(f'[checkpoint cleanup] Only {len(checkpoint_dirs)} checkpoints, no cleanup needed')
                return

            checkpoint_dirs.sort(
                key=lambda d: int(d.split('-')[1]) if d.split('-')[1].isdigit() else 0,
                reverse=True)

            protected_dir = None
            if protected_checkpoint_path:
                protected_dir = os.path.basename(protected_checkpoint_path)
                if protected_dir in checkpoint_dirs:
                    if protected_dir in checkpoint_dirs[keep_count:]:
                        checkpoint_dirs.remove(protected_dir)
                        checkpoint_dirs.sort(
                            key=lambda d: int(d.split('-')[1]) if d.split('-')[1].isdigit() else 0,
                            reverse=True)

            to_delete = checkpoint_dirs[keep_count:]
            for checkpoint_dir in to_delete:
                checkpoint_path = os.path.join(lora_dir, checkpoint_dir)
                try:
                    shutil.rmtree(checkpoint_path)
                    pass
                except Exception:
                    pass
        except Exception:
            pass

    @app.post('/v1/finetuneTasks')
    async def create_job(self, job: _JobDescription, finetune_task_id: str = Query(None),  # noqa B008
                         token: str = Header(DEFAULT_TOKEN)):  # noqa B008
        if not self._in_user_job_info(token):
            self._update_user_job_info(token)

        if finetune_task_id:
            user_jobs = self._read_user_job_info(token)
            for existing_job_id, existing_job_info in user_jobs.items():
                if existing_job_info.get('finetune_task_id') == finetune_task_id:
                    return {'finetune_task_id': existing_job_id, 'status': existing_job_info.get('status', 'Unknown')}

        # Build Job-ID:
        create_time = datetime.now().strftime(self._time_format)
        job_id = '-'.join(['ft', create_time, str(uuid.uuid4())[:5]])

        # Build Model-ID:
        characters = string.ascii_letters + string.digits
        random_string = ''.join(random.choices(characters, k=7))
        model_id = job.name + '_' + random_string

        # Build checkpoint save dir:
        # - No-Env-Set: (work/path + save_ckpt) + token + job_id;
        # - Env-Set:    (train_target_root)     + token + job_id;
        save_root = os.path.join(lazyllm.config['train_target_root'], token, job_id)
        os.makedirs(save_root, exist_ok=True)

        hypram = job.training_args.model_dump()

        stage = None
        if job.stage and job.stage.strip():
            stage = job.stage.lower().strip()
            supported_stages = ['sft', 'pt', 'dpo']
            if stage not in supported_stages:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f'Unsupported training stage: {stage}. '
                        f'Only supported stages are: {", ".join(supported_stages)}')
                )
            hypram['stage'] = stage

        assert len(job.training_dataset) == 1, 'just support one train dataset'
        data_path = job.training_dataset[0].dataset_download_uri
        if is_url(data_path):
            try:
                file_name = get_filename_from_url(data_path)
                target_path = os.path.join(save_root, file_name)
                response = requests.get(data_path, stream=True, timeout=300)
                if response.status_code == 200:
                    with open(target_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    data_path = target_path
                else:
                    raise HTTPException(status_code=404, detail='dataset download failed')
            except Exception as e:
                raise HTTPException(status_code=404, detail=f'dataset download failed: {str(e)}')

        if not stage or stage == 'sft':
            data_path = uniform_sft_dataset(data_path, target='alpaca')

        ngpus = 1
        if 'ngpus' in hypram:
            num_gpus = hypram['ngpus']
            if isinstance(num_gpus, str):
                try:
                    ngpus = int(num_gpus)
                except (ValueError, TypeError):
                    pass
            elif isinstance(num_gpus, int):
                ngpus = num_gpus
            hypram.pop('ngpus', None)

        if 'quantization_bit' in hypram:
            qbit = hypram['quantization_bit']
            if isinstance(qbit, str) and qbit.lower() not in ['null', 'none', '']:
                try:
                    hypram['quantization_bit'] = int(qbit)
                except (ValueError, TypeError):
                    pass
        if 'double_quantization' in hypram:
            dq = hypram['double_quantization']
            if isinstance(dq, str):
                if dq.lower() in ['true', '1', 'yes']:
                    hypram['double_quantization'] = True
                elif dq.lower() in ['false', '0', 'no', 'null', 'none']:
                    hypram['double_quantization'] = False

        if 'ref_model' in hypram:
            if not os.path.exists(hypram['ref_model']):
                default_path = os.path.join(lazyllm.config['model_path'], hypram['ref_model'])
                if os.path.exists(default_path):
                    hypram['ref_model'] = default_path
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f'ref_model {hypram["ref_model"]} not found')
        if 'reward_model' in hypram:
            if not os.path.exists(hypram['reward_model']):
                default_path = os.path.join(lazyllm.config['model_path'], hypram['reward_model'])
                if os.path.exists(default_path):
                    hypram['reward_model'] = default_path
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f'reward_model {hypram["reward_model"]} not found')

        if 'save_steps' not in hypram:
            hypram['save_steps'] = 500
        else:
            try:
                hypram['save_steps'] = int(hypram['save_steps'])
            except (ValueError, TypeError):
                hypram['save_steps'] = 500

        if 'save_steps' in hypram and hypram['save_steps'] > 0:
            self._start_periodic_checkpoint_cleanup(token, job_id)

        m = lazyllm.TrainableModule(job.model, save_root)\
            .trainset(data_path)\
            .finetune_method(lazyllm.finetune.auto)

        logger.info(
            f'[create_job] hypram: {hypram}, ngpus: {ngpus}, '
            f'model_id: {model_id}, data_path: {data_path}, save_root: {save_root}')
        thread = threading.Thread(target=m._impl._async_finetune, args=(model_id, ngpus), kwargs=hypram)
        thread.start()
        await asyncio.sleep(1)

        try:
            async with timeout(5):
                while m.status(model_id) == Status.Cancelled:
                    await asyncio.sleep(1)
        except asyncio.TimeoutError:
            pass

        save_path = self._get_save_path(m)
        log_path = m.log_path(model_id)

        status = m.status(model_id).name
        if Status[status] == Status.Running:
            started_time = datetime.now().strftime(self._time_format)
        else:
            started_time = None
        self._update_active_jobs(token, job_id, (m, thread))
        self._update_user_job_info(token, job_id, {
            'model_id': model_id,
            'finetune_task_id': finetune_task_id or job_id,
            'base_model': job.model,
            'created_at': create_time,
            'fine_tuned_model': save_path,
            'status': status,
            'data_path': data_path,
            'hyperparameters': hypram,
            'log_path': log_path,
            'started_at': started_time,
            'cost': None,
            'ngpus': ngpus,  # Store ngpus in job_info for resume_job to use
        })

        return {'finetune_task_id': job_id, 'status': status}

    @app.delete('/v1/finetuneTasks/{job_id}')
    async def cancel_job(self, job_id: str, token: str = Header(DEFAULT_TOKEN)):  # noqa B008
        if not self._in_user_job_info(token):
            self._update_user_job_info(token)
        await self.authorize_current_user(token)

        if self._in_active_jobs(token, job_id):
            try:
                await self.pause_job(job_id, token)
            except Exception as e:
                raise HTTPException(status_code=404, detail=f'Task {job_id}, cancelled failed, {e}')
            m, _ = self._pop_active_job(token, job_id)
            info = self._read_user_job_info(token, job_id)
            return {'status': m.status(info['model_id']).name}

        if self._in_user_job_info(token, job_id):
            info = self._read_user_job_info(token, job_id)
            status = info.get('status', '')

            if status in ('Done', 'Failed', 'Cancelled', 'Suspended'):
                try:
                    self._pop_user_job_info(token, job_id)
                    return {'status': status}
                except Exception as e:
                    logger.info(f'[cancel_job] Failed to remove job {job_id}: {e}')
                    raise HTTPException(
                        status_code=500, detail=f'Failed to remove job {job_id}: {e}')
            else:
                raise HTTPException(status_code=400, detail=f'Job {job_id} is in unexpected status: {status}')

        raise HTTPException(status_code=404, detail='Job not found')

    @app.get('/v1/finetuneTasks/jobs')
    async def list_jobs(self, token: str = Header(DEFAULT_TOKEN)):  # noqa B008
        # await self.authorize_current_user(token)
        if not self._in_user_job_info(token):
            self._update_user_job_info(token)
        save_root = os.path.join(lazyllm.config['train_target_root'], token)
        server_running_dict = self._read_user_job_info(token)
        m = lazyllm.TrainableModule('', save_root)
        valid_models, invalid_models = m.get_all_models()
        updated_jobs = {}

        # Collect all completed fine-tuning tasks from file system
        # (only process valid_models, i.e., completed fine-tuning)
        file_system_job_ids = set()
        for model_id, model_path in valid_models:
            job_id = model_path[len(save_root):].lstrip(os.sep).split(os.sep)[0]
            file_system_job_ids.add(job_id)

            if job_id in server_running_dict:
                # If the job in server_running_dict is not completed, mark it as Done
                if server_running_dict[job_id]['status'] != 'Done':
                    server_running_dict[job_id]['status'] = 'Done'
                    server_running_dict[job_id]['fine_tuned_model'] = model_path
                    updated_jobs[job_id] = server_running_dict[job_id]
                else:
                    server_running_dict[job_id]['fine_tuned_model'] = model_path
                    updated_jobs[job_id] = server_running_dict[job_id]
            else:
                server_running_dict[job_id] = {
                    'status': 'Done',
                    'model_id': model_id,
                    'fine_tuned_model': model_path,
                }
                updated_jobs[job_id] = server_running_dict[job_id]

        # Handle invalid_models (lazyllm_merge directory exists but no model files)
        for model_id, model_path in invalid_models:
            job_id = model_path[len(save_root):].lstrip(os.sep).split(os.sep)[0]
            file_system_job_ids.add(job_id)

            if job_id in server_running_dict:
                current_status = server_running_dict[job_id].get('status', '')
                if current_status == 'Done':
                    server_running_dict[job_id]['status'] = 'Failed'
                    server_running_dict[job_id]['fine_tuned_model'] = model_path
                    updated_jobs[job_id] = server_running_dict[job_id]
            else:
                server_running_dict[job_id] = {
                    'status': 'Failed',
                    'model_id': model_id,
                    'fine_tuned_model': model_path,
                }
                updated_jobs[job_id] = server_running_dict[job_id]

        jobs_to_remove = []
        for job_id, job_info in server_running_dict.items():
            if job_id not in file_system_job_ids:
                status = job_info.get('status', '')
                if status in ('Done', 'Failed', 'Cancelled'):
                    jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            try:
                await self.cancel_job(job_id, token)
            except Exception:
                try:
                    self._pop_user_job_info(token, job_id)
                except Exception:
                    pass
            finally:
                if job_id in server_running_dict:
                    del server_running_dict[job_id]

        # Clean up fine-tuning task folders created more than 1 month ago
        try:
            one_month_ago = datetime.now().timestamp() - 30 * 24 * 60 * 60  # Timestamp from 30 days ago
            if os.path.exists(save_root):
                for item in os.listdir(save_root):
                    item_path = os.path.join(save_root, item)
                    if os.path.isdir(item_path):
                        try:
                            mtime = os.path.getmtime(item_path)
                            if mtime < one_month_ago:
                                if item.startswith('ft-'):
                                    shutil.rmtree(item_path)
                                    pass
                        except Exception:
                            pass
        except Exception:
            pass

        for job_id, job_info in updated_jobs.items():
            self._update_user_job_info(token, job_id, job_info)

        return server_running_dict

    @app.get('/v1/finetuneTasks/{job_id}')
    async def get_job_info(self, job_id: str, token: str = Header(DEFAULT_TOKEN)):  # noqa B008
        if not self._in_user_job_info(token):
            self._update_user_job_info(token)
        await self.authorize_current_user(token)
        if not self._in_user_job_info(token, job_id):
            raise HTTPException(status_code=404, detail='Job not found')
        self._update_status(token, job_id)
        return self._read_user_job_info(token, job_id)

    @app.get('/v1/finetuneTasks/{job_id}/log')
    async def get_job_log(self, job_id: str, token: str = Header(DEFAULT_TOKEN)):  # noqa B008
        await self.authorize_current_user(token)
        if not self._in_user_job_info(token, job_id):
            raise HTTPException(status_code=404, detail='Job not found')

        self._update_status(token, job_id)
        info = self._read_user_job_info(token, job_id)

        if not info['log_path'] or not os.path.exists(info['log_path']):
            raise HTTPException(status_code=404, detail='log file not found')

        async def generate_log_stream():
            with open(info['log_path'], 'r') as f:
                for line in f:
                    if line.strip():
                        res = json.dumps({'result': {'log_data': line.strip()}})
                        yield f'data: {res}\n\n'
            yield 'data: [DONE]'

        return StreamingResponse(generate_log_stream(), media_type='text/event-stream')

    @app.post('/v1/finetuneTasks/{job_id}/model:export')
    async def export_model(self, job_id: str, model: ModelExport, token: str = Header(DEFAULT_TOKEN)):  # noqa B008
        if not self._in_user_job_info(token, job_id):
            raise HTTPException(status_code=404, detail='Job not found')

        self._update_status(token, job_id)
        info = self._read_user_job_info(token, job_id)

        model_path = info['fine_tuned_model']

        if model_path and 'lazyllm_lora' in model_path:
            merge_path = model_path.replace('lazyllm_lora', 'lazyllm_merge')
            if os.path.exists(merge_path):
                model_path = merge_path

        if not model_path or not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f'model file not found: {model_path}')

        target_dir = os.path.join(lazyllm.config['model_path'], model.model_display_name)
        if os.path.exists(target_dir):
            raise HTTPException(status_code=409, detail='target dir already exists')

        try:
            shutil.copytree(model_path, target_dir)
            logger.info(
                f'[export_model] Job {job_id} exported model to '
                f'{model.model_display_name}')
        except OSError as e:
            if 'No space left on device' in str(e):
                raise HTTPException(status_code=507, detail='Insufficient storage space for export')
            raise HTTPException(status_code=500, detail=f'Failed to copy model: {str(e)}')

        return

    @app.get('/v1/finetuneTasks/{job_id}/runningMetrics')
    async def get_running_metrics(self, job_id: str, token: str = Header(DEFAULT_TOKEN)):  # noqa B008
        raise HTTPException(status_code=404, detail='not implemented')

    @app.post('/v1/finetuneTasks/{job_id}:pause')
    async def pause_job(self, job_id: str, name: str = Body(embed=True), token: str = Header(DEFAULT_TOKEN)):  # noqa B008
        await self.authorize_current_user(token)
        self._update_status(token, job_id)
        if not self._in_active_jobs(token, job_id):
            raise HTTPException(status_code=404, detail='Job is ended or not found')

        m, _ = self._read_active_job(token, job_id)
        info = self._read_user_job_info(token, job_id)
        m.stop(info['model_id'])
        total_sleep = 0
        while m.status(info['model_id']) != Status.Cancelled:
            time.sleep(1)
            total_sleep += 1
            if total_sleep > 10:
                logger.info(f'[pause_job] Job {job_id} pause timed out')
                raise HTTPException(status_code=500, detail=f'Task {job_id}, pause timed out.')

        checkpoint_path = None
        lora_dir = self._get_lora_dir(info.get('fine_tuned_model'))
        if lora_dir and os.path.exists(lora_dir):
            finetuning_type = info.get('hyperparameters', {}).get('finetuning_type', 'lora')

            checkpoint_path, latest_incomplete_path = self._get_complete_checkpoint(
                lora_dir, finetuning_type=finetuning_type)

            # Delete incomplete latest checkpoint if:
            # 1. We found an older complete checkpoint (use it instead), OR
            # 2. No complete checkpoint found at all (delete to save space, will restart from scratch on resume)
            if latest_incomplete_path:
                try:
                    shutil.rmtree(latest_incomplete_path)
                except Exception:
                    pass

            # Only handle adapter_config.json for LoRA/QLoRA finetuning
            # Full finetuning doesn't use adapter, so skip this logic
            if finetuning_type in ('lora', 'qlora'):
                adapter_config_path = os.path.join(lora_dir, 'adapter_config.json')
                checkpoint_adapter = (os.path.join(checkpoint_path, 'adapter_config.json')
                                     if checkpoint_path else None)
                if (not os.path.exists(adapter_config_path) and
                        checkpoint_adapter and os.path.exists(checkpoint_adapter)):
                    shutil.copy(checkpoint_adapter, adapter_config_path)

            self._cleanup_old_checkpoints(lora_dir, keep_count=3, protected_checkpoint_path=checkpoint_path)

        cost = info.get('cost')
        if cost is None and info.get('started_at'):
            started_at = datetime.strptime(info['started_at'], self._time_format)
            current_segment_cost = (datetime.now() - started_at).total_seconds()
            cost = current_segment_cost

        # Extract checkpoint step before saving
        checkpoint_step = None
        if checkpoint_path:
            checkpoint_step = self._extract_checkpoint_step(checkpoint_path)
            if checkpoint_step is not None:
                logger.info(f'[pause_job] Job {job_id} extracted checkpoint step: {checkpoint_step}')

        self._update_user_job_info(token, job_id, {
            'status': 'Suspended',
            'checkpoint_path': checkpoint_path,
            'checkpoint_step': checkpoint_step,
            'cost': cost,
            'started_at': None,
            'last_cost_update_time': None,
        })

        try:
            self._pop_active_job(token, job_id)
        except Exception:
            pass
        return {'status': 'Suspended', 'checkpoint_path': checkpoint_path or '', 'cost': cost}

    @app.post('/v1/finetuneTasks/{job_id}:resume')
    async def resume_job(self, job_id: str, name: str = _DEFAULT_BODY_EMBED,
                         checkpoint_path: str = _DEFAULT_BODY_OPTIONAL,
                         token: str = _DEFAULT_HEADER_TOKEN):
        await self.authorize_current_user(token)
        if not self._in_user_job_info(token, job_id):
            raise HTTPException(status_code=404, detail='Job not found')

        info = self._read_user_job_info(token, job_id)
        if info.get('status') != 'Suspended':
            raise HTTPException(
                status_code=400,
                detail=f'Job is not suspended (current status: {info.get("status")})')

        if not checkpoint_path:
            checkpoint_path = info.get('checkpoint_path')

        has_checkpoint = checkpoint_path and os.path.exists(checkpoint_path)
        # Track whether this is a resume or restart
        resume_type = 'resume'  # Default to resume

        base_model = info.get('base_model')
        hyperparameters = info.get('hyperparameters', {}).copy()
        data_path = info.get('data_path')
        if not base_model or not data_path:
            raise HTTPException(status_code=400, detail='Missing required information for resume')

        if has_checkpoint:
            lora_dir = self._get_lora_dir(info.get('fine_tuned_model'))
            if lora_dir and os.path.exists(lora_dir):
                save_root = os.path.dirname(lora_dir)
            else:
                save_root = os.path.join(lazyllm.config['train_target_root'], token, job_id)
        else:
            # No checkpoint: start fresh with new save_root (similar to create_job)
            logger.info(f'[resume_job] Job {job_id} no checkpoint found, starting fresh training')
            resume_type = 'restart'
            save_root = os.path.join(lazyllm.config['train_target_root'], token, job_id)

        os.makedirs(save_root, exist_ok=True)

        m = lazyllm.TrainableModule(base_model, save_root)\
            .trainset(data_path)\
            .finetune_method(lazyllm.finetune.auto)

        if has_checkpoint:
            lora_dir = self._get_lora_dir(info.get('fine_tuned_model'))
            if lora_dir and os.path.exists(lora_dir):
                m._impl._specific_target_path = lora_dir

        model_id = info.get('model_id')
        if not model_id:
            characters = string.ascii_letters + string.digits
            random_string = ''.join(random.choices(characters, k=7))
            model_id = job_id + '_' + random_string

        if has_checkpoint:
            finetuning_type = hyperparameters.get('finetuning_type', 'lora')
            if not self._is_checkpoint_complete(checkpoint_path, finetuning_type):
                logger.info(f'[resume_job] Job {job_id} checkpoint incomplete, starting fresh training')
                resume_type = 'restart'
                has_checkpoint = False
                checkpoint_path = None
                # Remove resume_from_checkpoint from hyperparameters
                hyperparameters.pop('resume_from_checkpoint', None)
            else:
                removed_files = []

                # Delete rng_state.pth (single GPU training)
                rng_state_file = os.path.join(checkpoint_path, 'rng_state.pth')
                if os.path.exists(rng_state_file):
                    try:
                        os.remove(rng_state_file)
                        removed_files.append(os.path.basename(rng_state_file))
                    except Exception:
                        pass

                # Delete rng_state_*.pth files (multi-GPU training)
                rng_state_pattern = os.path.join(checkpoint_path, 'rng_state_*.pth')
                for rng_state_file in glob.glob(rng_state_pattern):
                    try:
                        os.remove(rng_state_file)
                        removed_files.append(os.path.basename(rng_state_file))
                    except Exception:
                        pass

                hyperparameters['resume_from_checkpoint'] = checkpoint_path
        else:
            hyperparameters.pop('resume_from_checkpoint', None)

        if 'save_steps' in hyperparameters and hyperparameters['save_steps'] > 0:
            self._start_periodic_checkpoint_cleanup(token, job_id)

        ngpus = info.get('ngpus', 1)
        if isinstance(ngpus, str):
            try:
                ngpus = int(ngpus)
            except (ValueError, TypeError):
                ngpus = 1
        elif not isinstance(ngpus, int):
            ngpus = 1

        thread = threading.Thread(target=m._impl._async_finetune, args=(model_id, ngpus), kwargs=hyperparameters)
        thread.start()
        await asyncio.sleep(1)

        try:
            async with timeout(5):
                while m.status(model_id) == Status.Cancelled:
                    await asyncio.sleep(1)
        except asyncio.TimeoutError:
            pass

        # The first getting the path may be invalid, and it will be getted with each update.
        save_path = self._get_save_path(m)
        log_path = m.log_path(model_id)
        # Save status
        status = m.status(model_id).name
        if Status[status] == Status.Running:
            started_time = datetime.now().strftime(self._time_format)
        else:
            started_time = None

        previous_cost = info.get('cost', 0) or 0

        if self._in_active_jobs(token, job_id):
            try:
                old_m, old_thread = self._read_active_job(token, job_id)
                try:
                    old_m.stop(info.get('model_id', ''))
                except Exception as e:
                    logger.info(f'[resume_job] Failed to stop old training for job {job_id}: {e}')
                self._pop_active_job(token, job_id)
            except Exception as e:
                logger.info(f'[resume_job] Failed to clean up old active_job entry for {job_id}: {e}')

        self._update_active_jobs(token, job_id, (m, thread))
        # Extract checkpoint step before clearing checkpoint_path
        checkpoint_step = None
        if has_checkpoint and checkpoint_path:
            checkpoint_step = self._extract_checkpoint_step(checkpoint_path)
            if checkpoint_step is not None:
                logger.info(f'[resume_job] Job {job_id} extracted checkpoint step: {checkpoint_step}')

        update_data = {
            'status': status,
            'model_id': model_id,
            'fine_tuned_model': save_path,
            'log_path': log_path,
            'started_at': started_time,
            'checkpoint_path': None,
            'cost': previous_cost,
            'last_cost_update_time': None,
            'checkpoint_step': checkpoint_step,
            'ngpus': ngpus,
            'hyperparameters': hyperparameters,
        }

        self._update_user_job_info(token, job_id, update_data)
        logger.info(
            f'[resume_job] Job {job_id} resumed successfully, '
            f'status: {status}, checkpoint: {has_checkpoint}, resume_type: {resume_type}')
        return {'status': status, 'resume_type': resume_type}

import os
import time
import uuid
import string
import random
import asyncio
import json
import threading
import logging
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
from lazyllm import FastapiApp as app
from lazyllm.tools.services import ServerBase

logger = logging.getLogger(__name__)

DEFAULT_TOKEN = 'default_token'

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
        except Exception as e:
            logger.warning(f"[_update_status] Failed to get status for job {job_id}: {e}")
            # If we can't get status, check if thread is still alive
            # If thread is dead and no status available, mark as Failed
            try:
                _, thread = self._read_active_job(token, job_id)
                if thread and not thread.is_alive():
                    # Thread is dead, likely due to exception during initialization
                    logger.warning(f"[_update_status] Thread for job {job_id} is dead, marking as Failed")
                    update = {'status': 'Failed'}
                    self._update_user_job_info(token, job_id, update)
                    # Pop from active jobs
                    try:
                        m, _ = self._pop_active_job(token, job_id)
                    except Exception:
                        pass
            except Exception:
                pass
            return

        update = {'status': status}

        # Some tasks not run when they are just created
        if Status[status] == Status.Running and not info.get('started_at'):
            update['started_at'] = datetime.now().strftime(self._time_format)

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

        if Status[status] == Status.Running and info.get('started_at'):
            now = datetime.now()
            started_at = datetime.strptime(info['started_at'], self._time_format)
            
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
                logger.warning(f"[_update_status] Job {job_id} progress: {progress_info}, cost: {total_cost}s, status: {status}")
            else:
                # If no progress from log, try to initialize from checkpoint step
                checkpoint_step = info.get('checkpoint_step')
                if checkpoint_step is not None and isinstance(checkpoint_step, int):
                    # Try to get total steps from log (if available) to calculate progress
                    log_path = info.get('log_path')
                    if log_path and os.path.exists(log_path):
                        try:
                            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                                lines = f.readlines()
                                total_steps = None
                                # First, try to find "Total optimization steps" line (most accurate)
                                for line in reversed(lines):
                                    if 'Total optimization steps' in line:
                                        total_steps_match = re.search(r'Total optimization steps\s*=\s*(\d+)', line)
                                        if total_steps_match:
                                            total_steps = int(total_steps_match.group(1))
                                            break
                                
                                # If not found, try to find from progress bar (less accurate)
                                if total_steps is None:
                                    # Exclude non-training progress bars (same logic as _extract_training_progress)
                                    non_training_patterns = [
                                        r'Loading checkpoint shards',
                                        r'Converting format of dataset',
                                        r'Running tokenizer on dataset',
                                    ]
                                    for line in reversed(lines[-100:]):
                                        # Skip non-training progress bars
                                        is_non_training = any(re.search(pattern, line, re.IGNORECASE) for pattern in non_training_patterns)
                                        if is_non_training:
                                            continue
                                        
                                        # Match progress bar format: " 50/150 [08:11<00:33,  1.11s/it]"
                                        progress_match = re.search(r'(\d+)/(\d+)', line)
                                        if progress_match:
                                            potential_total = int(progress_match.group(2))
                                            # Only use if it's a reasonable total (>= checkpoint_step and <= 10000)
                                            if potential_total >= checkpoint_step and potential_total <= 10000:
                                                total_steps = potential_total
                                                break
                                
                                if total_steps and total_steps > 0:
                                    base_percent = int((checkpoint_step / total_steps) * 100)
                                    update['progress_percent'] = max(0, min(100, base_percent))
                                    logger.warning(f"[_update_status] Job {job_id} initialized progress from checkpoint step {checkpoint_step}/{total_steps} = {update['progress_percent']}%")
                        except Exception:
                            pass
                
                # Clear progress_percent if no progress info available
                if 'progress_percent' not in update:
                    if 'progress_percent' in info:
                        update['progress_percent'] = None
                    logger.warning(f"[_update_status] Job {job_id} updated cost: {total_cost}s (previous: {previous_cost}s, current segment: {current_segment_cost}s), status: {status}")

        # Update Status
        self._update_user_job_info(token, job_id, update)

        # Pop and kill jobs with status: Failed
        if Status[status] == Status.Failed:
            # Clear progress_percent when task is Failed
            update['progress_percent'] = None
            if 'cost' not in update:
                # If update doesn't contain cost, calculate final cost
                # Prefer using last_cost_update_time for incremental calculation to avoid double counting
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
                self._update_user_job_info(token, job_id, {'cost': total_cost, 'progress_percent': None})
                logger.warning(f"[_update_status] Job {job_id} final cost (Failed): {total_cost}s, cleared progress_percent")
            else:
                # update already has cost (accumulated from Running status), use it directly
                # Ensure progress_percent is cleared
                update['progress_percent'] = None
                logger.warning(f"[_update_status] Job {job_id} final cost (Failed, already updated): {update['cost']}s, cleared progress_percent")
            
            try:
                m, _ = self._pop_active_job(token, job_id)
                m.stop(info['model_id'])
            except Exception as e:
                logger.warning(f"[_update_status] Failed to stop job {job_id} after Failed status: {e}")
            return

        if Status[status] == Status.Done:
            # Check if training actually failed by examining log file for errors
            log_path = info.get('log_path') or update.get('log_path')
            should_mark_failed = False
            if log_path and os.path.exists(log_path):
                try:
                    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                        log_content = f.read()
                        # Check for common failure indicators
                        if any(error in log_content for error in [
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
                        ]):
                            # If error found but status is Done, change to Failed
                            logger.warning(f"[_update_status] Job {job_id} marked as Done but found errors in log, changing to Failed")
                            should_mark_failed = True
                except Exception as e:
                    logger.warning(f"[_update_status] Failed to check log file for job {job_id}: {e}")
            
            # If errors found, treat as Failed
            if should_mark_failed:
                update['status'] = 'Failed'
                status = 'Failed'
                # Update status immediately before calculating cost
                self._update_user_job_info(token, job_id, update)
                # Fall through to Failed handling
                if 'cost' not in update:
                    previous_cost = info.get('cost', 0) or 0
                    if info.get('last_cost_update_time') and info.get('started_at'):
                        last_update_time = datetime.strptime(info['last_cost_update_time'], self._time_format)
                        current_segment_cost = (datetime.now() - last_update_time).total_seconds()
                        total_cost = previous_cost + current_segment_cost
                    elif info.get('started_at'):
                        started_at = datetime.strptime(info['started_at'], self._time_format)
                        total_cost = (datetime.now() - started_at).total_seconds()
                    else:
                        total_cost = previous_cost
                    self._update_user_job_info(token, job_id, {'cost': total_cost})
                    logger.warning(f"[_update_status] Job {job_id} final cost (Failed): {total_cost}s")
                else:
                    logger.warning(f"[_update_status] Job {job_id} final cost (Failed, already updated): {update['cost']}s")
                
                try:
                    m, _ = self._pop_active_job(token, job_id)
                    m.stop(info['model_id'])
                except Exception as e:
                    logger.warning(f"[_update_status] Failed to stop job {job_id} after Failed status: {e}")
                return
            
            # Normal Done handling
            # Clear progress_percent when task is Done (completed)
            update['progress_percent'] = None
            if 'cost' not in update:
                # If update doesn't contain cost, calculate final cost
                # Prefer using last_cost_update_time for incremental calculation to avoid double counting
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
                self._update_user_job_info(token, job_id, {'cost': total_cost, 'progress_percent': None})
                logger.warning(f"[_update_status] Job {job_id} final cost (Done): {total_cost}s, cleared progress_percent")
            else:
                # update already has cost (accumulated from Running status), use it directly
                # Ensure progress_percent is cleared
                update['progress_percent'] = None
                logger.warning(f"[_update_status] Job {job_id} final cost (Done, already updated): {update['cost']}s, cleared progress_percent")
            
            try:
                m, _ = self._pop_active_job(token, job_id)
            except Exception as e:
                logger.warning(f"[_update_status] Failed to pop job {job_id} after Done status: {e}")
            return

        create_time = datetime.strptime(info['created_at'], self._time_format)
        delta_time = (datetime.now() - create_time).total_seconds()

        # More than 5 min pop and kill jobs with status: Cancelled. Because of
        # some tasks have just been started and their status cannot be checked.
        if delta_time > 300 and Status[status] == Status.Cancelled:
            m, _ = self._pop_active_job(token, job_id)
            m.stop(info['model_id'])
            if info.get('started_at') and not info.get('cost'):
                cost = (datetime.now() - datetime.strptime(info['started_at'], self._time_format)).total_seconds()
                self._update_user_job_info(token, job_id, {'cost': cost})
            return

        # More than 50 min pop and kill jobs with status: TBSubmitted, InQueue, Pending
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
                # Find "Running training" line to determine search range
                training_start_idx = None
                last_200_lines = lines[-200:]
                for idx, line in enumerate(last_200_lines):
                    if 'Running training' in line or 'Total optimization steps' in line:
                        training_start_idx = idx
                        break
                
                # Only search in lines after "Running training" (if found)
                if training_start_idx is not None:
                    search_lines = last_200_lines[training_start_idx + 1:]  # Lines after "Running training"
                else:
                    search_lines = last_200_lines  # If not found, search all lines
                
                # Search backwards for progress information
                for line in reversed(search_lines):  # Only check lines after training started
                    original_line = line
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Match progress bar format: " 75%|███████▌  | 90/120 [08:11<00:33,  1.11s/it]"
                    progress_match = re.search(r'(\d+)%\s*\|\s*[█▉▊▋▌▍▎▏\s]+\s*\|\s*(\d+)/(\d+)', line)
                    if progress_match:
                        # Exclude non-training progress bars with descriptive prefixes
                        non_training_patterns = [
                            r'Loading checkpoint shards',
                            r'Converting format of dataset',
                            r'Running tokenizer on dataset',
                        ]
                        is_non_training = any(re.search(pattern, original_line, re.IGNORECASE) for pattern in non_training_patterns)
                        
                        # Only extract training progress (has time format and no descriptive prefix)
                        if not is_non_training:
                            # Check if has time format (training progress bar has [HH:MM<HH:MM, X.XXs/it])
                            time_format_pattern = r'\[\d+:\d+<\d+:\d+,\s*[\d.]+\s*s/it\]'
                            if re.search(time_format_pattern, original_line):
                                percent = int(progress_match.group(1))
                                current_step = int(progress_match.group(2))
                                total_steps = int(progress_match.group(3))
                                
                                # Try to extract loss information
                                loss_match = re.search(r"'loss':\s*([\d.]+)", line)
                                loss = float(loss_match.group(1)) if loss_match else None
                                
                                # Try to extract epoch information
                                epoch_match = re.search(r"'epoch':\s*([\d.]+)", line)
                                epoch = float(epoch_match.group(1)) if epoch_match else None
                                
                                return {
                                    'percent': percent,
                                    'step': f"{current_step}/{total_steps}",
                                    'loss': loss,
                                    'epoch': epoch
                                }
                    
                    # Match training metrics format: "{'loss': 0.6341, 'epoch': 20.0, ...}"
                    metrics_match = re.search(r"\{'loss':\s*([\d.]+).*'epoch':\s*([\d.]+)", line)
                    if metrics_match:
                        loss = float(metrics_match.group(1))
                        epoch = float(metrics_match.group(2))
                        return {
                            'loss': loss,
                            'epoch': epoch
                        }
        except Exception as e:
            logger.debug(f"[_extract_training_progress] Failed to extract progress from {log_path}: {e}")
        
        return None

    def _extract_checkpoint_step(self, checkpoint_path):
        """Extract step number from checkpoint path (e.g., checkpoint-40 -> 40)"""
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
                trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
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
            
            checkpoint_dirs.sort(key=lambda d: int(d.split('-')[1]) if d.split('-')[1].isdigit() else 0, reverse=True)
            
            # Check the latest checkpoint first
            latest_path = os.path.join(lora_dir, checkpoint_dirs[0])
            if self._is_checkpoint_complete(latest_path, finetuning_type):
                return latest_path, None
            
            # Latest checkpoint is incomplete, return the second latest (which must be complete)
            if len(checkpoint_dirs) >= 2:
                second_latest_path = os.path.join(lora_dir, checkpoint_dirs[1])
                logger.warning(f"[_get_complete_checkpoint] Latest checkpoint {checkpoint_dirs[0]} is incomplete, using second latest {checkpoint_dirs[1]} (must be complete)")
                return second_latest_path, latest_path
            
            # Only one checkpoint and it's incomplete
            logger.warning(f"[_get_complete_checkpoint] Only one checkpoint {checkpoint_dirs[0]} exists and it's incomplete, returning None")
            return None, latest_path
        except Exception as e:
            logger.warning(f"[_get_complete_checkpoint] Error: {e}")
            return None, None

    def _get_lora_dir(self, fine_tuned_model):
        if not fine_tuned_model:
            return None
        if 'lazyllm_lora' in fine_tuned_model:
            return fine_tuned_model if fine_tuned_model.endswith('lazyllm_lora') else fine_tuned_model[:fine_tuned_model.find('lazyllm_lora') + len('lazyllm_lora')]
        elif 'lazyllm_merge' in fine_tuned_model:
            return fine_tuned_model.replace('lazyllm_merge', 'lazyllm_lora')
        else:
            return os.path.join(os.path.dirname(fine_tuned_model), 'lazyllm_lora')

    def _cleanup_old_checkpoints(self, lora_dir, keep_count=3, protected_checkpoint_path=None):
        if not lora_dir or not os.path.exists(lora_dir):
            logger.warning(f"[checkpoint cleanup] lora_dir does not exist: {lora_dir}")
            return
        
        try:
            checkpoint_dirs = [
                d for d in os.listdir(lora_dir)
                if d.startswith('checkpoint-') and os.path.isdir(os.path.join(lora_dir, d))
            ]
            if len(checkpoint_dirs) <= keep_count:
                logger.debug(f"[checkpoint cleanup] Only {len(checkpoint_dirs)} checkpoints, no cleanup needed")
                return
            
            checkpoint_dirs.sort(key=lambda d: int(d.split('-')[1]) if d.split('-')[1].isdigit() else 0, reverse=True)
            
            protected_dir = None
            if protected_checkpoint_path:
                protected_dir = os.path.basename(protected_checkpoint_path)
                if protected_dir in checkpoint_dirs:
                    if protected_dir in checkpoint_dirs[keep_count:]:
                        checkpoint_dirs.remove(protected_dir)
                        checkpoint_dirs.sort(key=lambda d: int(d.split('-')[1]) if d.split('-')[1].isdigit() else 0, reverse=True)
                        logger.warning(f"[checkpoint cleanup] Protected checkpoint {protected_dir} from deletion")
            
            to_delete = checkpoint_dirs[keep_count:]
            for checkpoint_dir in to_delete:
                checkpoint_path = os.path.join(lora_dir, checkpoint_dir)
                try:
                    shutil.rmtree(checkpoint_path)
                    logger.warning(f"[checkpoint cleanup] Deleted old checkpoint: {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"[checkpoint cleanup] Failed to delete checkpoint {checkpoint_path}: {e}")
            
            logger.warning(f"[checkpoint cleanup] Cleaned up {len(to_delete)} old checkpoints, kept {keep_count} latest")
        except Exception as e:
            logger.warning(f"[checkpoint cleanup] Error during cleanup: {e}")

    @app.post('/v1/finetuneTasks')
    async def create_job(self, job: _JobDescription, finetune_task_id: str = Query(None),  # noqa B008
                         token: str = Header(DEFAULT_TOKEN)):  # noqa B008
        if not self._in_user_job_info(token):
            self._update_user_job_info(token)
        logger.warning(f"[create_job] Job: {job.model_dump_json()}")

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

        # Add launcher into hyperparameters:
        hypram = job.training_args.model_dump()
        
        # Add stage parameter if provided (for PT, RM, PPO, DPO training modes)
        # Only add if stage is not empty to avoid overriding default 'sft' from sft.yaml
        if job.stage and job.stage.strip():
            hypram['stage'] = job.stage.lower().strip()
        
        # Uniform Training DataSet:
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

        data_path = uniform_sft_dataset(data_path, target='alpaca')

        ngpus = 1
        if 'ngpus' in hypram:
            num_gpus = hypram['ngpus']
            if isinstance(num_gpus, str):
                try:
                    num_gpus = int(num_gpus)
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

        if 'save_steps' not in hypram:
            hypram['save_steps'] = 100
        else:
            try:
                hypram['save_steps'] = int(hypram['save_steps'])
            except (ValueError, TypeError):
                hypram['save_steps'] = 100

        cleanup_thread = None
        if 'save_steps' in hypram and hypram['save_steps'] > 0:
            def periodic_cleanup():
                time.sleep(30)
                while True:
                    try:
                        time.sleep(60)
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
                            self._cleanup_old_checkpoints(lora_dir, keep_count=3, protected_checkpoint_path=protected_checkpoint)
                    except Exception as e:
                        logger.warning(f"[checkpoint cleanup] Error in periodic cleanup thread for job {job_id}: {e}")
                        time.sleep(60)
            
            cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
            cleanup_thread.start()
            logger.warning(f"[create_job] Started checkpoint cleanup thread for job {job_id} (save_steps={hypram['save_steps']})")

        m = lazyllm.TrainableModule(job.model, save_root)\
            .trainset(data_path)\
            .finetune_method(lazyllm.finetune.auto)

        # Launch Training:
        thread = threading.Thread(target=m._impl._async_finetune, args=(model_id,ngpus), kwargs=hypram)
        thread.start()
        await asyncio.sleep(1)

        # Sleep 5s for launch cmd.
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
        })

        return {'finetune_task_id': job_id, 'status': status}

    @app.delete('/v1/finetuneTasks/{job_id}')
    async def cancel_job(self, job_id: str, token: str = Header(DEFAULT_TOKEN)):  # noqa B008
        if not self._in_user_job_info(token):
            self._update_user_job_info(token)
        await self.authorize_current_user(token)
        
        # Check if the job is in _active_jobs (running tasks)
        if self._in_active_jobs(token, job_id):
            # Job is running, need to pause first
            try:
                await self.pause_job(job_id, token)
            except Exception as e:
                raise HTTPException(status_code=404, detail=f'Task {job_id}, cancelled failed, {e}')
            m, _ = self._pop_active_job(token, job_id)
            info = self._read_user_job_info(token, job_id)
            return {'status': m.status(info['model_id']).name}
        
        # Job is not in _active_jobs, might be a completed task
        # Check if it's in _user_job_info
        if self._in_user_job_info(token, job_id):
            info = self._read_user_job_info(token, job_id)
            status = info.get('status', '')
            
            # If job is completed (Done/Failed/Cancelled), remove the job record directly
            if status in ('Done', 'Failed', 'Cancelled', 'Suspended'):
                try:
                    self._pop_user_job_info(token, job_id)
                    logger.warning(f"[cancel_job] Removed completed job {job_id} from user_job_info (status: {status})")
                    return {'status': status}
                except Exception as e:
                    logger.warning(f"[cancel_job] Failed to remove job {job_id} from user_job_info: {e}")
                    raise HTTPException(status_code=500, detail=f'Failed to remove job {job_id}: {e}')
            else:
                # Job status is abnormal, return error
                raise HTTPException(status_code=400, detail=f'Job {job_id} is in unexpected status: {status}')
        
        # Job not found
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
        
        # Collect all completed fine-tuning tasks from file system (only process valid_models, i.e., completed fine-tuning)
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
                    # Even if status is already Done, update fine_tuned_model path to ensure it's always up-to-date
                    server_running_dict[job_id]['fine_tuned_model'] = model_path
                    updated_jobs[job_id] = server_running_dict[job_id]
            else:
                # If not in server_running_dict, add it to server_running_dict
                server_running_dict[job_id] = {
                    'status': 'Done',
                    'model_id': model_id,
                    'fine_tuned_model': model_path,
                }
                updated_jobs[job_id] = server_running_dict[job_id]
        
        # Handle invalid_models (lazyllm_merge directory exists but no model files)
        for model_id, model_path in invalid_models:
            job_id = model_path[len(save_root):].lstrip(os.sep).split(os.sep)[0]
            file_system_job_ids.add(job_id)  # Add to set to avoid being removed
            
            if job_id in server_running_dict:
                # If the job is in server_running_dict, check its status
                current_status = server_running_dict[job_id].get('status', '')
                # If status is 'Done' but no model files exist, status is inconsistent, should change to 'Failed'
                if current_status == 'Done':
                    server_running_dict[job_id]['status'] = 'Failed'
                    server_running_dict[job_id]['fine_tuned_model'] = model_path
                    updated_jobs[job_id] = server_running_dict[job_id]
                # If status is 'Running', 'Pending' etc. (running states), keep unchanged
                # because it might still be training and model files haven't been generated yet
                # If status is already 'Failed', also keep unchanged
            else:
                # If not in server_running_dict, add the job info to server_running_dict with status Failed
                server_running_dict[job_id] = {
                    'status': 'Failed',
                    'model_id': model_id,
                    'fine_tuned_model': model_path,
                }
                updated_jobs[job_id] = server_running_dict[job_id]
        
        # Remove extra tasks from server_running_dict (not found in file system)
        # But keep tasks with status Running, Pending, InQueue etc. (running tasks)
        jobs_to_remove = []
        for job_id, job_info in server_running_dict.items():
            if job_id not in file_system_job_ids:
                status = job_info.get('status', '')
                # Only remove completed or failed tasks, keep running tasks
                if status in ('Done', 'Failed', 'Cancelled'):
                    jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            try:
                # Call cancel_job to cleanup tasks, continue regardless of success/failure to prevent memory leaks
                await self.cancel_job(job_id, token)
                logger.warning(f"[list_jobs] Cleaned up job {job_id} via cancel_job (not found in file system)")
            except Exception as e:
                # Catch all exceptions (including HTTPException) to ensure cleanup process continues
                logger.warning(f"[list_jobs] Failed to cleanup job {job_id} via cancel_job: {e}, removing from server_running_dict anyway")
                # Even if cancel_job fails, try to remove from server_running_dict to prevent memory leaks
                try:
                    self._pop_user_job_info(token, job_id)
                except Exception:
                    pass
                if job_id in server_running_dict:
                    del server_running_dict[job_id]
        
        # Clean up fine-tuning task folders created more than 1 month ago
        try:
            one_month_ago = datetime.now().timestamp() - 30 * 24 * 60 * 60  # Timestamp from 30 days ago
            if os.path.exists(save_root):
                for item in os.listdir(save_root):
                    item_path = os.path.join(save_root, item)
                    if os.path.isdir(item_path):
                        # Check directory creation time
                        try:
                            # Use modification time as approximation of creation time
                            mtime = os.path.getmtime(item_path)
                            if mtime < one_month_ago:
                                # Check if it's a fine-tuning task directory (starts with ft-)
                                if item.startswith('ft-'):
                                    shutil.rmtree(item_path)
                                    logger.warning(f"[list_jobs] Removed old training task directory: {item_path} (older than 1 month)")
                        except Exception as e:
                            logger.warning(f"[list_jobs] Failed to check/remove directory {item_path}: {e}")
        except Exception as e:
            logger.warning(f"[list_jobs] Failed to cleanup old training directories: {e}")
        
        # Update all changed task information
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
                logger.warning(f"[export_model] Job {job_id} using merge_path: {model_path}")
        
        if not model_path or not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f'model file not found: {model_path}')
        
        target_dir = os.path.join(lazyllm.config['model_path'], model.model_display_name)
        if os.path.exists(target_dir):
            raise HTTPException(status_code=409, detail='target dir already exists')
        
        try:
            shutil.copytree(model_path, target_dir)
            logger.warning(f"[export_model] Job {job_id} copied model_path {model_path} to target_dir: {target_dir}")
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
        logger.warning(f"[pause_job] Starting pause for job {job_id}, token={token}")
        await self.authorize_current_user(token)
        self._update_status(token, job_id)
        if not self._in_active_jobs(token, job_id):
            logger.warning(f"[pause_job] Job {job_id} not found in active_jobs")
            raise HTTPException(status_code=404, detail='Job is ended or not found')

        m, _ = self._read_active_job(token, job_id)
        info = self._read_user_job_info(token, job_id)
        logger.warning(f"[pause_job] Job {job_id} current status: {info.get('status')}, model_id: {info.get('model_id')}")
        
        logger.warning(f"[pause_job] Stopping training for job {job_id}")
        m.stop(info['model_id'])
        total_sleep = 0
        while m.status(info['model_id']) != Status.Cancelled:
            time.sleep(1)
            total_sleep += 1
            if total_sleep > 10:
                logger.warning(f"[pause_job] Job {job_id} pause timed out after {total_sleep} seconds")
                raise HTTPException(status_code=500, detail=f'Task {job_id}, pause timed out.')
        
        logger.warning(f"[pause_job] Job {job_id} stopped successfully after {total_sleep} seconds")

        checkpoint_path = None
        lora_dir = self._get_lora_dir(info.get('fine_tuned_model'))
        logger.warning(f"[pause_job] Job {job_id} lora_dir: {lora_dir}")
        if lora_dir and os.path.exists(lora_dir):
            # Get finetuning_type from hyperparameters to determine checkpoint validation method
            finetuning_type = info.get('hyperparameters', {}).get('finetuning_type', 'lora')
            logger.warning(f"[pause_job] Job {job_id} finetuning_type: {finetuning_type}")
            
            # Get complete checkpoint (if latest is incomplete, use older complete one)
            checkpoint_path, latest_incomplete_path = self._get_complete_checkpoint(lora_dir, finetuning_type=finetuning_type)
            
            # Delete incomplete latest checkpoint if:
            # 1. We found an older complete checkpoint (use it instead), OR
            # 2. No complete checkpoint found at all (delete to save space, will restart from scratch on resume)
            if latest_incomplete_path:
                try:
                    shutil.rmtree(latest_incomplete_path)
                    if checkpoint_path:
                        logger.warning(f"[pause_job] Job {job_id} deleted incomplete latest checkpoint: {os.path.basename(latest_incomplete_path)} (found older complete checkpoint)")
                    else:
                        logger.warning(f"[pause_job] Job {job_id} deleted incomplete latest checkpoint: {os.path.basename(latest_incomplete_path)} (no complete checkpoint found, will restart from scratch on resume)")
                except Exception as e:
                    logger.warning(f"[pause_job] Job {job_id} failed to delete incomplete checkpoint {latest_incomplete_path}: {e}")
            
            if checkpoint_path:
                logger.warning(f"[pause_job] Job {job_id} found complete checkpoint: {checkpoint_path}")
            else:
                logger.warning(f"[pause_job] Job {job_id} no complete checkpoint found (this is normal if training hasn't reached save_steps yet). Resume will restart from scratch.")
            
            # Only handle adapter_config.json for LoRA/QLoRA finetuning
            # Full finetuning doesn't use adapter, so skip this logic
            if finetuning_type in ('lora', 'qlora'):
                adapter_config_path = os.path.join(lora_dir, 'adapter_config.json')
                if not os.path.exists(adapter_config_path):
                    if checkpoint_path and os.path.exists(os.path.join(checkpoint_path, 'adapter_config.json')):
                        shutil.copy(os.path.join(checkpoint_path, 'adapter_config.json'), adapter_config_path)
                        logger.warning(f"[pause_job] Job {job_id} copied adapter_config.json from checkpoint to lora_dir")
                    else:
                        logger.warning(f"[pause_job] Job {job_id} adapter_config.json not found (no checkpoint available)")
                else:
                    logger.warning(f"[pause_job] Job {job_id} adapter_config.json already exists in lora_dir")
            else:
                logger.warning(f"[pause_job] Job {job_id} finetuning_type is {finetuning_type}, skipping adapter_config.json handling (not needed for Full finetuning)")
            
            # During pause, protect the checkpoint we're returning (if any)
            self._cleanup_old_checkpoints(lora_dir, keep_count=3, protected_checkpoint_path=checkpoint_path)
        else:
            logger.warning(f"[pause_job] Job {job_id} lora_dir does not exist or is invalid: {lora_dir} (no checkpoint available, resume will restart from scratch)")

        # Note: _update_status has already updated cost in real-time (previous_cost + current_segment_cost)
        # So here we just use the latest cost value, no need to recalculate
        # If _update_status hasn't updated cost yet (e.g., task just started), calculate manually
        cost = info.get('cost')
        if cost is None and info.get('started_at'):
            # Only calculate when cost doesn't exist (backward compatible)
            current_segment_cost = (datetime.now() - datetime.strptime(info['started_at'], self._time_format)).total_seconds()
            cost = current_segment_cost
            logger.warning(f"[pause_job] Job {job_id} calculated cost (fallback): {cost} seconds")
        else:
            logger.warning(f"[pause_job] Job {job_id} using cost from _update_status: {cost} seconds")

        # Always update status to Suspended, regardless of checkpoint availability
        # No checkpoint is a valid state - resume will simply restart from scratch
        self._update_user_job_info(token, job_id, {
            'status': 'Suspended',
            'checkpoint_path': checkpoint_path,
            'cost': cost,
            'started_at': None,
            'last_cost_update_time': None,  # Clear last update time
        })
        
        try:
            self._pop_active_job(token, job_id)
            logger.warning(f"[pause_job] Job {job_id} removed from active_jobs")
        except Exception as e:
            logger.warning(f"[pause_job] Failed to remove job {job_id} from active_jobs: {e}")
        
        # Pause is successful even without checkpoint - resume will restart from scratch if needed
        if checkpoint_path:
            logger.warning(f"[pause_job] Job {job_id} paused successfully with checkpoint: {checkpoint_path}, cost: {cost}s")
        else:
            logger.warning(f"[pause_job] Job {job_id} paused successfully without checkpoint (resume will restart from scratch), cost: {cost}s")
        return {'status': 'Suspended', 'checkpoint_path': checkpoint_path or '', 'cost': cost}

    @app.post('/v1/finetuneTasks/{job_id}:resume')
    async def resume_job(self, job_id: str, name: str = Body(embed=True),
                         checkpoint_path: str = Body(None, embed=True),
                         token: str = Header(DEFAULT_TOKEN)):  # noqa B008
        logger.warning(f"[resume_job] Starting resume for job {job_id}, token={token}, checkpoint_path={checkpoint_path}")
        await self.authorize_current_user(token)
        if not self._in_user_job_info(token, job_id):
            logger.warning(f"[resume_job] Job {job_id} not found in user_job_info")
            raise HTTPException(status_code=404, detail='Job not found')
        
        info = self._read_user_job_info(token, job_id)
        logger.warning(f"[resume_job] Job {job_id} current status: {info.get('status')}")
        if info.get('status') != 'Suspended':
            logger.warning(f"[resume_job] Job {job_id} is not suspended, cannot resume")
            raise HTTPException(status_code=400, detail=f'Job is not suspended (current status: {info.get("status")})')
        
        if not checkpoint_path:
            checkpoint_path = info.get('checkpoint_path')
            logger.warning(f"[resume_job] Job {job_id} using checkpoint_path from info: {checkpoint_path}")
        
        has_checkpoint = checkpoint_path and os.path.exists(checkpoint_path)
        logger.warning(f"[resume_job] Job {job_id} has_checkpoint: {has_checkpoint}, checkpoint_path: {checkpoint_path}")
        
        base_model = info.get('base_model')
        hyperparameters = info.get('hyperparameters', {}).copy()
        data_path = info.get('data_path')
        logger.warning(f"[resume_job] Job {job_id} base_model: {base_model}, data_path: {data_path}")
        if not base_model or not data_path:
            logger.warning(f"[resume_job] Job {job_id} missing required information: base_model={base_model}, data_path={data_path}")
            raise HTTPException(status_code=400, detail='Missing required information for resume')
        
        if has_checkpoint:
            lora_dir = self._get_lora_dir(info.get('fine_tuned_model'))
            if lora_dir and os.path.exists(lora_dir):
                save_root = os.path.dirname(lora_dir)
            else:
                save_root = os.path.join(lazyllm.config['train_target_root'], token, job_id)
            logger.warning(f"[resume_job] Job {job_id} using existing save_root: {save_root}")
        else:
            # No checkpoint: start fresh with new save_root (similar to create_job)
            save_root = os.path.join(lazyllm.config['train_target_root'], token, job_id)
            logger.warning(f"[resume_job] Job {job_id} creating new save_root: {save_root}")
        
        os.makedirs(save_root, exist_ok=True)
        
        m = lazyllm.TrainableModule(base_model, save_root)\
            .trainset(data_path)\
            .finetune_method(lazyllm.finetune.auto)
        
        if has_checkpoint:
            lora_dir = self._get_lora_dir(info.get('fine_tuned_model'))
            if lora_dir and os.path.exists(lora_dir):
                m._impl._specific_target_path = lora_dir
                logger.warning(f"[resume_job] Job {job_id} set _specific_target_path: {lora_dir}")
        
        model_id = info.get('model_id')
        if not model_id:
            characters = string.ascii_letters + string.digits
            random_string = ''.join(random.choices(characters, k=7))
            model_id = job_id + '_' + random_string
            logger.warning(f"[resume_job] Job {job_id} generated new model_id: {model_id}")
        else:
            logger.warning(f"[resume_job] Job {job_id} using existing model_id: {model_id}")
        
        if has_checkpoint:
            # Validate checkpoint completeness before resume
            finetuning_type = hyperparameters.get('finetuning_type', 'lora')
            if not self._is_checkpoint_complete(checkpoint_path, finetuning_type):
                # Checkpoint is incomplete, treat as no checkpoint and restart training
                logger.warning(f"[resume_job] Job {job_id} checkpoint is incomplete: {checkpoint_path}, treating as no checkpoint and restarting training from scratch")
                has_checkpoint = False
                checkpoint_path = None
                # Remove resume_from_checkpoint from hyperparameters
                hyperparameters.pop('resume_from_checkpoint', None)
            else:
                # Checkpoint is complete, proceed with resume
                # Delete all rng_state files to avoid weights_only=True loading issue
                # RNG state files contain numpy objects, which causes UnpicklingError when resuming training
                # For multi-GPU training, each GPU has its own rng_state_N.pth file (e.g., rng_state_0.pth, rng_state_1.pth)
                removed_files = []
                
                # Delete rng_state.pth (single GPU training)
                rng_state_file = os.path.join(checkpoint_path, "rng_state.pth")
                if os.path.exists(rng_state_file):
                    try:
                        os.remove(rng_state_file)
                        removed_files.append(os.path.basename(rng_state_file))
                        logger.warning(f"[resume_job] Job {job_id} removed RNG state file: {os.path.basename(rng_state_file)}")
                    except Exception as e:
                        logger.warning(f"[resume_job] Job {job_id} failed to remove RNG state file {rng_state_file}: {e}")
                
                # Delete rng_state_*.pth files (multi-GPU training)
                rng_state_pattern = os.path.join(checkpoint_path, "rng_state_*.pth")
                for rng_state_file in glob.glob(rng_state_pattern):
                    try:
                        os.remove(rng_state_file)
                        removed_files.append(os.path.basename(rng_state_file))
                        logger.warning(f"[resume_job] Job {job_id} removed RNG state file: {os.path.basename(rng_state_file)}")
                    except Exception as e:
                        logger.warning(f"[resume_job] Job {job_id} failed to remove RNG state file {rng_state_file}: {e}")
                
                if removed_files:
                    logger.warning(f"[resume_job] Job {job_id} removed {len(removed_files)} RNG state file(s): {', '.join(removed_files)}")
                
                hyperparameters['resume_from_checkpoint'] = checkpoint_path
                logger.warning(f"[resume_job] Job {job_id} set resume_from_checkpoint: {checkpoint_path}")
        else:
            hyperparameters.pop('resume_from_checkpoint', None)
            logger.warning(f"[resume_job] Job {job_id} no checkpoint available, will restart training from scratch (this is normal if pause occurred before first save_steps)")
        
        if 'save_steps' in hyperparameters and hyperparameters['save_steps'] > 0:
            def periodic_cleanup():
                time.sleep(30)  # Wait for the first checkpoint to appear
                while True:
                    try:
                        time.sleep(60)  # Check every minute
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
                            self._cleanup_old_checkpoints(lora_dir, keep_count=3, protected_checkpoint_path=protected_checkpoint)
                    except Exception as e:
                        logger.warning(f"[checkpoint cleanup] Error in periodic cleanup thread for job {job_id}: {e}")
                        time.sleep(60)
            
            cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
            cleanup_thread.start()
            logger.warning(f"[resume_job] Started checkpoint cleanup thread for job {job_id} (save_steps={hyperparameters['save_steps']})")
        
        logger.warning(f"[resume_job] Job {job_id} starting training thread with model_id: {model_id}")
        thread = threading.Thread(target=m._impl._async_finetune, args=(model_id,), kwargs=hyperparameters)
        thread.start()
        await asyncio.sleep(1)
        
        # Sleep 5s for launch cmd.
        try:
            async with timeout(5):
                while m.status(model_id) == Status.Cancelled:
                    await asyncio.sleep(1)
        except asyncio.TimeoutError:
            pass
        
        # The first getting the path may be invalid, and it will be getted with each update.
        save_path = self._get_save_path(m)
        log_path = m.log_path(model_id)
        logger.warning(f"[resume_job] Job {job_id} save_path: {save_path}, log_path: {log_path}")
        
        # Save status
        status = m.status(model_id).name
        logger.warning(f"[resume_job] Job {job_id} resumed with status: {status}")
        if Status[status] == Status.Running:
            started_time = datetime.now().strftime(self._time_format)
        else:
            started_time = None

        previous_cost = info.get('cost', 0) or 0
        logger.warning(f"[resume_job] Job {job_id} preserving previous cost: {previous_cost}s")

        if self._in_active_jobs(token, job_id):
            try:
                old_m, old_thread = self._read_active_job(token, job_id)
                logger.warning(f"[resume_job] Job {job_id} already in active_jobs, cleaning up old entry")
                try:
                    old_m.stop(info.get('model_id', ''))
                except Exception as e:
                    logger.warning(f"[resume_job] Failed to stop old training for job {job_id}: {e}")
                self._pop_active_job(token, job_id)
            except Exception as e:
                logger.warning(f"[resume_job] Failed to clean up old active_job entry for {job_id}: {e}")

        self._update_active_jobs(token, job_id, (m, thread))
        # Extract checkpoint step before clearing checkpoint_path
        checkpoint_step = None
        if has_checkpoint and checkpoint_path:
            checkpoint_step = self._extract_checkpoint_step(checkpoint_path)
            if checkpoint_step is not None:
                logger.warning(f"[resume_job] Job {job_id} extracted checkpoint step: {checkpoint_step}")
        
        update_data = {
            'status': status,
            'model_id': model_id,
            'fine_tuned_model': save_path,
            'log_path': log_path,
            'started_at': started_time,
            'checkpoint_path': None,
            'cost': previous_cost,
            'last_cost_update_time': None,
            'checkpoint_step': checkpoint_step,  # Save for progress initialization
            'hyperparameters': hyperparameters,  # Save updated hyperparameters (including resume_from_checkpoint) so cleanup thread can read it
        }
        
        self._update_user_job_info(token, job_id, update_data)
        logger.warning(f"[resume_job] Job {job_id} resumed successfully, preserved cost: {previous_cost}s")
        return {'status': status}
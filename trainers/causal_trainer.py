import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pandas as pd
from models.base_model import sequence_collate_fn

class CausalTrainer:
    def __init__(self, model, train_dataset, val_dataset, config, local_rank, global_rank, world_size):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = config
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.device = torch.device("cuda", local_rank)
        
        # Centralized path tracking anchored to the dynamic experiment identifier
        self.checkpoint_path = f"checkpoints/{config['run_name']}.pt"
        self.log_path = f"results/{config['run_name']}/metrics.csv"
        self.inference_log_path = f"results/{config['run_name']}/inference_stats.txt"
        
        self.start_step = 1
        self.accumulated_train_time = 0.0
        self.peak_vram = 0.0
        
        self.metrics = {
            "step": [], "train_loss": [], "val_loss": [], 
            "val_perplexity": [], "lr": [], 
            "accumulated_time": [], "train_tokens_per_sec": [], "peak_vram_gb": []
        }
        
    def prepare_infrastructure(self):
        if os.path.exists(self.checkpoint_path):
            if self.global_rank == 0:
                print(f"Detected existing checkpoint state. Resuming pipeline cleanly...")
            checkpoint = torch.load(self.checkpoint_path, map_location=f"cuda:{self.local_rank}")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_step = checkpoint['step'] + 1
            self.accumulated_train_time = checkpoint.get('accumulated_train_time', 0.0)
            
            if self.global_rank == 0 and os.path.exists(self.log_path):
                try:
                    df_existing = pd.read_csv(self.log_path)
                    df_existing = df_existing[df_existing['step'] < self.start_step]
                    self.metrics = df_existing.to_dict(orient='list')
                except Exception as e:
                    print(f"Warning: Metric stream sync skipped: {e}")
            return checkpoint
        return None

    @torch.no_grad()
    def evaluate(self, eval_steps=15):
        self.model.eval()
        total_loss = 0.0
        steps = 0
        
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            self.val_dataset, num_replicas=self.world_size, rank=self.global_rank, shuffle=False
        )
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.cfg["batch_size"], sampler=val_sampler, 
            num_workers=4, pin_memory=True, collate_fn=sequence_collate_fn
        )
        
        for x, y in val_loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            steps += 1
            if steps >= eval_steps:
                break
                
        loss_tensor = torch.tensor(total_loss, device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / (steps * self.world_size)
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
        
        self.model.train()
        return avg_loss, perplexity

    @torch.no_grad()
    def benchmark_inference(self, prompt_len=32, gen_len=2048):
        self.model.eval()
        raw_model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        
        # Enforce fused recurrent optimizations on external hardware layers if present
        for block in raw_model.layers:
            if hasattr(block, 'attn') and hasattr(block.attn, 'mode'):
                try:
                    block.attn.mode = 'fused_recurrent'
                except Exception:
                    pass
                
        # -------------------------------------------------------------------------
        # ROUTE TRACK 1: Native Framework Decoupled Recurrent Step Engines
        # -------------------------------------------------------------------------
        if self.cfg["model_type"] in ["transformer", "cs_lrad", "ccrs", "ckts", "cbkc", "cofe"]:
            prompt_ids = torch.randint(0, 20000, (1, prompt_len), device=self.device)
            past_states = None
            
            # Execute prompt pre-fill phase to initialize internal sequence memory structures
            for t in range(prompt_len):
                token_step = prompt_ids[:, t:t+1]
                _, past_states = raw_model.step_inference(token_step, past_states=past_states)
                
            h_input = prompt_ids[:, -1:]
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            # Unroll sequential generation to measure pure generation throughput
            for _ in range(gen_len):
                logits, past_states = raw_model.step_inference(h_input, past_states=past_states)
                h_input = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                
            torch.cuda.synchronize()
            return gen_len / (time.perf_counter() - start_time)
        
        # -------------------------------------------------------------------------
        # ROUTE TRACK 2: External Triton-Fused Layer State Engines
        # -------------------------------------------------------------------------
        elif self.cfg["model_type"] in ["gla", "gdn2", "mamba3"]:
            prompt_ids = torch.randint(0, 20000, (1, prompt_len), device=self.device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                h = raw_model.token_embedding(prompt_ids)
                past_states = []
                for block in raw_model.layers:
                    h_attn = block.ln1(h)
                    outputs = block.attn(h_attn, initial_state=None, output_final_state=True)
                    attn_out = outputs[0]
                    final_state = outputs[1] if len(outputs) == 2 else outputs[-1]
                    h = h + attn_out
                    h = h + block.mlp(block.ln2(h))
                    past_states.append(final_state)
                logits = raw_model.lm_head(raw_model.final_norm(h))
                current_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            # Step-by-step unroll leveraging fast parallel hardware state handoffs
            for _ in range(gen_len):
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    h = raw_model.token_embedding(current_token)
                    next_states = []
                    for i, block in enumerate(raw_model.layers):
                        h_attn = block.ln1(h)
                        outputs = block.attn(h_attn, initial_state=past_states[i], output_final_state=True)
                        attn_out = outputs[0]
                        final_state = outputs[1] if len(outputs) == 2 else outputs[-1]
                        h = h + attn_out
                        h = h + block.mlp(block.ln2(h))
                        next_states.append(final_state)
                    past_states = next_states
                    logits = raw_model.lm_head(raw_model.final_norm(h))
                    current_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    
            torch.cuda.synchronize()
            return gen_len / (time.perf_counter() - start_time)
        
        else:
            raise NotImplementedError(f"No optimized inference routing defined for model type: {self.cfg['model_type']}")

    def train(self):
        checkpoint_state = self.prepare_infrastructure()
        
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[self.local_rank], output_device=self.local_rank
        )
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])
        
        if checkpoint_state is not None:
            optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
            del checkpoint_state
            
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_dataset, num_replicas=self.world_size, rank=self.global_rank, shuffle=True, seed=42
        )
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.cfg["batch_size"], sampler=train_sampler, 
            num_workers=4, pin_memory=True, collate_fn=sequence_collate_fn
        )
        
        torch.cuda.reset_peak_memory_stats()
        step = self.start_step
        start_micro_step = (self.start_step - 1) * self.cfg["grad_accum"]
        micro_step_count = 0
        running_train_loss = 0.0
        optimizer.zero_grad()
        accumulated_loss = 0.0
        
        # --- PRE-COMPUTE EXTENDED MULTI-EPOCH RESUME BOUNDARIES ---
        steps_per_epoch = len(train_sampler)
        target_epoch = start_micro_step // steps_per_epoch
        epoch_skip_steps = start_micro_step % steps_per_epoch
        
        if self.global_rank == 0:
            step_start_time = time.perf_counter()
            
        for epoch in range(5):
            if step > self.cfg["max_steps"]: break
            
            if epoch < target_epoch:
                micro_step_count += steps_per_epoch
                continue
                
            if epoch == target_epoch and epoch_skip_steps > 0:
                if self.global_rank == 0:
                    print(f"Stateful sampler active: precision skipping first {epoch_skip_steps} rows of Epoch {epoch}...")
                train_sampler.set_epoch(epoch)
                all_indices = list(train_sampler)
                remaining_indices = all_indices[epoch_skip_steps:]
                
                current_loader = torch.utils.data.DataLoader(
                    self.train_dataset, batch_size=self.cfg["batch_size"], sampler=remaining_indices, 
                    num_workers=4, pin_memory=True, collate_fn=sequence_collate_fn
                )
                micro_step_count += epoch_skip_steps
            else:
                train_sampler.set_epoch(epoch)
                current_loader = train_loader
            
            for x, y in current_loader:
                if step > self.cfg["max_steps"]: break
                    
                # Dynamic learning rate scheduling referencing configuration
                warmup = self.cfg["warmup_steps"]
                if step < warmup:
                    lr = self.cfg["lr"] * (step / warmup)
                else:
                    progress = (step - warmup) / (self.cfg["max_steps"] - warmup)
                    lr = (self.cfg["lr"] * 0.1) + 0.5 * (self.cfg["lr"] - (self.cfg["lr"] * 0.1)) * (1.0 + math.cos(math.pi * progress))
                    
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                
                if (micro_step_count + 1) % self.cfg["grad_accum"] != 0:
                    with ddp_model.no_sync():
                        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                            logits = ddp_model(x)
                            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)) / self.cfg["grad_accum"]
                        loss.backward()
                else:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits = ddp_model(x)
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)) / self.cfg["grad_accum"]
                    loss.backward()
                    
                accumulated_loss += loss.item() * self.cfg["grad_accum"]
                micro_step_count += 1
                
                if micro_step_count % self.cfg["grad_accum"] == 0:
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    running_train_loss += accumulated_loss
                    accumulated_loss = 0.0
                    
                    if step % 500 == 0:
                        torch.cuda.synchronize()
                        val_loss, val_ppl = self.evaluate()
                        
                        if self.global_rank == 0:
                            chunk_duration = time.perf_counter() - step_start_time
                            self.accumulated_train_time += chunk_duration
                            
                            global_batch = self.cfg["batch_size"] * self.world_size
                            train_tokens_per_sec = (500 * global_batch * self.cfg["seq_len"]) / chunk_duration
                            self.peak_vram = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                            avg_train_loss = running_train_loss / (500 * self.cfg["grad_accum"])
                            
                            print(f"{step:<8} | {avg_train_loss:<12.4f} | {val_loss:<12.4f} | {val_ppl:<12.2f} | {lr:.6f} | {self.accumulated_train_time:<12.1f} | {train_tokens_per_sec:<12.1f}")
                            
                            self.metrics["step"].append(step)
                            self.metrics["train_loss"].append(avg_train_loss)
                            self.metrics["val_loss"].append(val_loss)
                            self.metrics["val_perplexity"].append(val_ppl)
                            self.metrics["lr"].append(lr)
                            self.metrics["accumulated_time"].append(self.accumulated_train_time)
                            self.metrics["train_tokens_per_sec"].append(train_tokens_per_sec)
                            self.metrics["peak_vram_gb"].append(self.peak_vram)
                            
                            pd.DataFrame(self.metrics).to_csv(self.log_path, index=False)
                            
                            tmp_path = f"{self.checkpoint_path}.tmp"
                            torch.save({
                                'step': step, 'accumulated_train_time': self.accumulated_train_time,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                            }, tmp_path)
                            os.replace(tmp_path, self.checkpoint_path)
                            step_start_time = time.perf_counter()
                            
                        dist.barrier()
                        running_train_loss = 0.0
                    step += 1
                
        dist.barrier()
        if self.global_rank == 0:
            print("\nTraining loop run complete. Launching isolated Inference Throughput evaluation...")
            if self.peak_vram == 0.0:
                self.peak_vram = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                
        inf_speed = self.benchmark_inference(prompt_len=32, gen_len=self.cfg["seq_len"])
        inf_tensor = torch.tensor(inf_speed, device=self.device)
        dist.all_reduce(inf_tensor, op=dist.ReduceOp.SUM)
        avg_inf_speed = inf_tensor.item() / self.world_size
        
        if self.global_rank == 0:
            print("="*85)
            print("FINAL GENERATED HARDWARE PERFORMANCE PROFILE")
            print(f"Total Logged Training Duration: {self.accumulated_train_time:.2f} Sec")
            print(f"Isolated Inference Speed: {avg_inf_speed:.2f} Tokens/Sec")
            print("="*85 + "\n")
            
            with open(self.inference_log_path, "w") as f:
                f.write(f"Architecture: {self.cfg['run_name']}\n")
                f.write(f"Total Trainable Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}\n")
                f.write(f"Total Training Wall Time (Sec): {self.accumulated_train_time:.2f}\n")
                f.write(f"Inference Tokens/Sec: {avg_inf_speed:.2f}\n")
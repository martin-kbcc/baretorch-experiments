import torch
import torch.nn.functional as F
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from transformers import AutoTokenizer
from models import make_model

@register_model("baretorch")
class BareTorchEvalWrapper(LM):
    def __init__(self, model_type, checkpoint_path, d_model=256, num_heads=4, num_layers=8, device="cuda", batch_size=32, **kwargs):
        super().__init__()
        self._device = torch.device(device)
        self._batch_size = int(batch_size) 
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.vocab_size = len(self.tokenizer)
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        self.pad_id = self.tokenizer.pad_token_id
        
        # 1. Rebuild the exact model shape from our factory
        self.model = make_model(
            model_type, 
            vocab_size=self.vocab_size, 
            d_model=int(d_model), 
            num_heads=int(num_heads), 
            num_layers=int(num_layers),
            **kwargs
        ).to(self._device)
        
        # 2. Bind thelegacy trained weights cleanly
        print(f"Loading BareTorch weights for validation from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

    @property
    def max_length(self):
        return 1024 

    @property
    def batch_size(self):
        return self._batch_size 

    @property
    def device(self):
        return self._device

    def loglikelihood(self, requests):
        """
        Calculates the conditional log-probabilities of target completions using 
        a streamlined, high-speed Left-Padded Batching loop.
        """
        results = []
        b_size = self.batch_size
        
        with torch.no_grad():
            for chunk_idx in range(0, len(requests), b_size):
                batch_requests = requests[chunk_idx : chunk_idx + b_size]
                
                batch_ctx_ids = []
                batch_cont_ids = []
                batch_full_tokens = []
                
                # Step 1: Extract and concatenate tokens
                for req in batch_requests:
                    context, continuation = req.args
                    ctx_ids = self.tokenizer.encode(context, add_special_tokens=False)
                    cont_ids = self.tokenizer.encode(continuation, add_special_tokens=False)
                    
                    batch_ctx_ids.append(ctx_ids)
                    batch_cont_ids.append(cont_ids)
                    batch_full_tokens.append(ctx_ids + cont_ids)
                
                max_len = max(len(tokens) for tokens in batch_full_tokens)
                current_batch_len = len(batch_requests)
                
                # Step 2: Build the uniform rectangular tensor grid
                input_tensor = torch.full(
                    (current_batch_len, max_len), 
                    fill_value=self.pad_id, 
                    dtype=torch.long, 
                    device=self._device
                )
                
                # Left-pad every row so text always terminates perfectly at the right edge
                for i, tokens in enumerate(batch_full_tokens):
                    pad_len = max_len - len(tokens)
                    input_tensor[i, pad_len:] = torch.tensor(tokens, device=self._device)
                
                # Step 3: Run parallel batch forward pass
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = self.model(input_tensor)
                
                # Step 4: Extract log-probabilities via absolute forward alignments
                for i in range(current_batch_len):
                    cont_len = len(batch_cont_ids[i])
                    cont_ids = batch_cont_ids[i]
                    
                    # Because text is right-aligned, the target logits are cleanly mapped
                    start_idx = max_len - cont_len - 1
                    end_idx = max_len - 1
                    
                    row_logits = logits[i, start_idx:end_idx, :]
                    row_targets = torch.tensor(cont_ids, device=self._device)
                    
                    log_probs = F.log_softmax(row_logits, dim=-1)
                    target_log_probs = log_probs[torch.arange(cont_len), row_targets]
                    
                    total_logprob = target_log_probs.sum().item()
                    is_greedy = (row_logits.argmax(dim=-1) == row_targets).all().item()
                    
                    results.append((total_logprob, is_greedy))
                    
        return results

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("LAMBADA execution configuration does not use rolling evaluations.")

    def generate_until(self, requests):
        raise NotImplementedError("LAMBADA execution configuration does not use generative evaluations.")
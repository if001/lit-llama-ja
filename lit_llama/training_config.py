import json
import math

class TrainingConfig():
    def __init__(self,
                model_size,
                learning_rate,
                min_lr,
                batch_size,
                micro_batch_size,
                max_iters,        
                weight_decay,
                beta1,
                beta2,
                grad_clip,
                decay_lr,
                warmup_iters,
                lr_decay_iters
                ) -> None:
        self.model_size = model_size
        self.learning_rate = learning_rate        
        self.min_lr = min_lr
        self.batch_size = batch_size        
        self.micro_batch_size = micro_batch_size
        self.max_iters = max_iters        
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.grad_clip = grad_clip
        self.decay_lr = decay_lr
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters

    def save(self, output_dir):
        """
        Save member variables of this instance to a JSON file.

        Parameters:
        - output_dir: The output dir of the JSON file to save to.
        """        
        member_vars = {k: v for k, v in self.__dict__.items() if not callable(v)}
        
        output_file = f'{output_dir}/training_config.json'
        with open(output_file, 'w') as f:
            json.dump(member_vars, f, ensure_ascii=False, indent=4)
        print(f'save training config... {output_file}')

    def debug(self):
        print('='*100)
        print('print training config...')
        for k, v in self.__dict__.items():
            print(f"{k}: {v}")
        print('='*100)

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    @classmethod
    def from_name(cls, model_size):
        if model_size == "Llama-2-250M-hf":
            max_iters = 253000
            conf = dict(
                model_size=model_size,
                learning_rate=9e-4,
                min_lr=9e-5,
                batch_size=160,
                micro_batch_size=4,
                max_iters=max_iters,
                weight_decay=0.01,
                beta1=0.9,
                beta2=0.95,
                grad_clip=2.0,
                decay_lr=True,
                warmup_iters=1000,
                lr_decay_iters=max_iters,
            )
            return cls(**conf)
        elif model_size == "Llama-2-130M-hf":
            # max_iters = 253000
            max_iters = 453000
            conf = dict(
                model_size=model_size,
                learning_rate=1e-3,
                min_lr=1e-4,
                batch_size=64,
                micro_batch_size=4,
                max_iters=max_iters,
                weight_decay=0.01,
                beta1=0.9,
                beta2=0.95,
                grad_clip=1.0,
                decay_lr=True,
                warmup_iters=500,
                lr_decay_iters=max_iters,
            )
            return cls(**conf)
        elif model_size == "Llama-2-350M-hf":
            max_iters = 253000
            conf = dict(
                model_size=model_size,
                learning_rate=1e-3,
                min_lr=1e-4,
                batch_size=128,
                micro_batch_size=4,
                max_iters=max_iters,
                weight_decay=0.01,
                beta1=0.9,
                beta2=0.95,
                grad_clip=1.0,
                decay_lr=True,
                warmup_iters=1000,
                lr_decay_iters=max_iters,
            )
            return cls(**conf)
        elif model_size == "Llama-2-350M_v2-hf":
            max_iters = 253000
            conf = dict(
                model_size=model_size,
                learning_rate=1e-3,
                min_lr=1e-4,
                batch_size=200,
                micro_batch_size=4,
                max_iters=max_iters,
                weight_decay=0.01,
                beta1=0.9,
                beta2=0.95,
                grad_clip=1.0,
                decay_lr=True,
                warmup_iters=1000,
                lr_decay_iters=max_iters,
            )
            return cls(**conf)
        elif model_size == "Llama-2-400M-hf":
            max_iters = 453000
            conf = dict(
                model_size=model_size,
                learning_rate=4e-4,
                min_lr=1e-5,
                batch_size=256,
                micro_batch_size=1,
                max_iters=max_iters,
                weight_decay=0.01,
                beta1=0.9,
                beta2=0.95,
                grad_clip=1.0,
                decay_lr=True,
                warmup_iters=1000,
                lr_decay_iters=max_iters,
            )
            return cls(**conf)
        elif model_size == "open_llama_130M":
            max_iters = 143000
            conf = dict(
                model_size=model_size,
                learning_rate = 0.0008,
                min_lr = 0.00008,
                batch_size = 128,                
                micro_batch_size = 4,
                max_iters = max_iters,
                weight_decay = 0.1,
                beta1 = 0.9,
                beta2 = 0.95,
                grad_clip = 1.0,
                decay_lr = True,
                warmup_iters = 2000,
                lr_decay_iters = max_iters,                
            )
            return cls(**conf)
        elif model_size == "phi-1_5-130M_v2":
            max_iters = 143000
            conf = dict(
                model_size=model_size,
                learning_rate = 0.0008,
                min_lr = 0.00008,
                batch_size = 128,                
                micro_batch_size = 4,
                max_iters = max_iters,
                weight_decay = 0.1,
                beta1 = 0.9,
                beta2 = 0.95,
                grad_clip = 1.0,
                decay_lr = True,
                warmup_iters = 2000,
                lr_decay_iters = max_iters,                
            )
            return cls(**conf)
        elif model_size == "148M":
            max_iters = 143000
            max_iters = 153000
            conf = dict(
                model_size="llama-148M",
                learning_rate = 0.0008,
                min_lr = 0.00008,
                batch_size = 128,                
                micro_batch_size = 4,
                max_iters = max_iters,
                weight_decay = 0.1,
                beta1 = 0.9,
                beta2 = 0.95,
                grad_clip = 1.0,
                decay_lr = True,
                warmup_iters = 2000,
                lr_decay_iters = max_iters,
            )
            return cls(**conf)
        else:
            raise ValueError("invalid model size", model_size)
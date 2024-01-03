import math
import json

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
                lr_decay_iters,
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
        if model_size == "Mixtral-100M":
            # block_size = 4096
            block_size = 960
            ds_size = 8e+9
            batch_size = 128
            micro_batch_size = 4
            one_iters = int(ds_size/(block_size*micro_batch_size))
            max_iters = one_iters
            conf = dict(
                model_size=model_size,
                learning_rate=1e-4,
                min_lr=1e-5,
                batch_size=batch_size,
                micro_batch_size=micro_batch_size,
                max_iters=max_iters,
                weight_decay=0.001,
                beta1=0.9,
                beta2=0.95,
                grad_clip=1.0,
                decay_lr=True,
                warmup_iters=500,
                lr_decay_iters=max_iters,
            )
            return cls(**conf)
        raise ValueError("invalid model size", model_size)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, 
                 embed_size=128, 
                 num_experts=4, 
                 expert_hidden_size=256, 
                 gate_hidden_size=256,
                 device="auto",
                 ):
        super(MoE, self).__init__()
        self.embed_size = embed_size
        self.num_experts = num_experts

        expert = nn.Sequential(
                nn.Linear(embed_size, expert_hidden_size),
                nn.ReLU(),
                nn.Linear(expert_hidden_size, embed_size)
            )
        
        self.experts = nn.ModuleList([expert for _ in range(num_experts)])

        self.gate =  nn.Sequential(
            nn.Linear(embed_size, gate_hidden_size),
            nn.ELU(),
            nn.Linear(gate_hidden_size, num_experts),
        )

    def forward(self, x):
        """
        Forward pass for MoE
        :param x: Input tensor of shape (batch_size, seq_len, embed_size)
        :return: Output tensor. shape (batch_size, seq_len, embed_size)
        """
        # batch_size, seq_len, _ = x.size()
        
        gating_scores = self.gate(x)
        gating_weights = F.softmax(gating_scores, dim=2) # (batch_size, seq_len, num_experts)        
        
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2) # (batch_size, seq_len, num_experts, embed_size)
        ## gating_weights.unsqueeze(-1) # (batch_size, seq_len, num_experts, 1)
        output = torch.sum(gating_weights.unsqueeze(-1) * expert_outputs, dim=2) #  (batch_size, seq_len, embed_size)        
        return output
    

class SparseMoE(nn.Module):
    def __init__(self, embed_size, num_experts, expert_hidden_size, top_k=1):
        super(SparseMoE, self).__init__()
        self.embed_size = embed_size
        self.num_experts = num_experts
        self.top_k = top_k

        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_size, expert_hidden_size),
                nn.ReLU(),
                nn.Linear(expert_hidden_size, embed_size)
            ) for _ in range(num_experts)
        ])

        # Gate
        self.gate = nn.Linear(embed_size, num_experts)

    def forward(self, x):
        """
        Forward pass for MoE
        :param x: Input tensor of shape (batch_size, seq_len, embed_size)
        :return: Output tensor. shape (batch_size, seq_len, embed_size)
        """
        # batch_size, seq_len, _ = x.size()

        gating_scores = self.gate(x)
        gating_probs = F.softmax(gating_scores, dim=2)        
        
        top_k_probs, top_k_indices = gating_probs.topk(self.top_k, dim=2)        

        output = torch.zeros_like(x)
        
        for expert_idx in range(self.num_experts):
            expert_mask = (top_k_indices == expert_idx).float().unsqueeze(3)
            expert_output = self.experts[expert_idx](x)
            expert_output = expert_output.unsqueeze(2)
            expert_contribution = expert_output * expert_mask
            output += expert_contribution.sum(2)
        return output, gating_scores

class MixtralBLockSparseTop2MLP(nn.Module):
    """
    https://github.com/huggingface/transformers/blob/v4.36.2/src/transformers/models/mixtral/modeling_mixtral.py#L688
    """
    def __init__(self, 
                 ffn_dim = 128,
                 hidden_dim = 128,
                 ):
        super().__init__()
        self.ffn_dim = ffn_dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = nn.ReLU()

    def forward(self, hidden_states, routing_weights):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return routing_weights * current_hidden_states
    
class MixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self,
                 num_experts = 2,
                 top_k = 1,
                 embed_size = 128,
                 expert_hidden_size = 128, 
                 gate_hidden_size = 128,
                 ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # self.gate = nn.Linear(embed_size, self.num_experts, bias=False)
        self.gate =  nn.Sequential(
            nn.Linear(embed_size, gate_hidden_size),
            nn.ELU(),
            nn.Linear(gate_hidden_size, num_experts),
        )
        self.experts = nn.ModuleList([MixtralBLockSparseTop2MLP(expert_hidden_size, embed_size) for _ in range(num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states) # (batch * sequence_length, n_experts)        
        
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float) # (batch * sequence_length, n_experts)

        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]

            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state, routing_weights[top_x_list, idx_list, None])
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

def get_load_balance_loss_0(
        gate_logits,
        top_k=2,
        num_experts=2
):
    """    
    v3.36.2ではバグがあるので注意
    
    以下のissueでmainには取り込まれている
    https://github.com/huggingface/transformers/issues/28093

    まだ壊れている。以下のPRの内容を取り込む
    https://github.com/huggingface/transformers/pull/28256
    """

    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    selected_experts = selected_experts.reshape(-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)        
    expert_mask = expert_mask.reshape(-1,top_k, num_experts)
    expert_mask = torch.max(expert_mask, dim=-2).values

    # Compute the percentage of tokens routed to each experts
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0) / top_k

    # Compute the average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    # overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(-1))
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert)
    return overall_loss * num_experts

def get_load_balance_loss(gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2) -> float:
    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)
    else:
        concatenated_gate_logits = gate_logits

    # routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
    routing_weights = torch.softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1) # [batch_size X sequence_length, top_k]

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts) # [batch_size X sequence_length, top_k, num_experts]
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0) # [top_k, num_experts]    

    # Compute the average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0) # [num_experts]
    # overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0)) # / top_k    
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0)) / top_k
    return overall_loss * num_experts

def router_z_loss(logits, num_microbatches, importance=None):
    """Loss that encourages router logits to remain small and improves stability.

    Args:
      logits: a tensor with shape [<batch_dims>, experts_dim]
      experts_dim: the number of experts (an integer)
      num_microbatches: number of microbatches
      importance: an optional tensor with shape [<batch_dims>, group_size_dim]

    Returns:
      z_loss: scalar loss only applied by non-padded tokens and normalized by
        num_microbatches.
    """
    if isinstance(logits, tuple):
        compute_device = logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in logits], dim=0)
    # Compute logsumexp across the experts dimension            
    log_z = torch.logsumexp(concatenated_gate_logits, dim=-1)    
    
    # Compute the square of log_z
    z_loss = torch.square(log_z)
    
    if importance is not None:
        # Apply importance weighting
        importance_mask = torch.eq(importance, 1.0).to(dtype=z_loss.dtype)
        z_loss *= importance_mask
        
        # Calculate the denominator for normalization
        denom = torch.sum(importance_mask)
    else:
        denom = concatenated_gate_logits.shape[-1]

    # Normalize the loss
    z_loss = torch.sum(z_loss) / (denom * num_microbatches)    
    return z_loss

def model_test():
    torch.set_default_dtype(torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 1
    seq_len = 3
    embding_size = 4
    expert_hidden_size= 5
    # moe = SparseMoE(
    #     embed_size=embding_size,
    #     num_experts=2,
    #     expert_hidden_size=expert_hidden_size
    # )
    moe = MixtralSparseMoeBlock(
        num_experts=3,
        num_experts_per_tok=2,
        embed_size=embding_size, 
        expert_hidden_size=expert_hidden_size
    )
    # moe = MoE(
    #     embed_size=embding_size,
    #     num_experts=2,
    #     expert_hidden_size=5
    # )
    inp = torch.rand([batch_size, seq_len, embding_size])
    print('input:', inp.shape, inp)
    out, _ = moe.forward(inp)
    print('output:', out.shape, out)

def loss_test():
    batch_size = 1
    seq_len = 100
    num_experts = 8
    layers = 2
    top_k = 2
    for _ in range(10):
        logits = [torch.randn([batch_size*seq_len, num_experts]) for _ in range(layers)]
        logits = tuple(logits)
        # print(logits)
        out = get_load_balance_loss2(logits, top_k=top_k, num_experts=num_experts)
        print(out)        
        # for logit in logits:
        #     out = get_load_balance_loss2(logit, top_k=top_k, num_experts=num_experts)
        #     print(out)

        # out = get_load_balance_loss(logits, top_k=top_k, num_experts=num_experts)
        # out = router_z_loss(logits, seq_len*batch_size)
        

def main():
    # model_test()
    loss_test()

if __name__ == '__main__':
    main()
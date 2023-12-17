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
        print('gating_weights', gating_weights.shape, gating_weights)
        
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2) # (batch_size, seq_len, num_experts, embed_size)
        ## gating_weights.unsqueeze(-1) # (batch_size, seq_len, num_experts, 1)
        output = torch.sum(gating_weights.unsqueeze(-1) * expert_outputs, dim=2) #  (batch_size, seq_len, embed_size)        
        return output
    
def main():
    torch.set_default_dtype(torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 1
    seq_len = 3
    embding_size = 4

    moe = MoE(
        embed_size=embding_size,
        num_experts=2,
        expert_hidden_size=5
    )
    inp = torch.rand([batch_size, seq_len, embding_size])


    print('input:', inp.shape, inp)
    out = moe.forward(inp)
    print('output:', out.shape, out)

if __name__ == '__main__':
    main()
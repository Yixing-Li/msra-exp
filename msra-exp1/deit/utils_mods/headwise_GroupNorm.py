import torch
import torch.nn as nn

class HeadWiseGroupNorm(nn.Module):
    def __init__(self, num_heads: int, num_groups: int, num_features: int, eps: float = 1e-7):
        """
        Headwise Group Normalization for multi-head attention mechanisms.
        
        Args:
            num_heads (int): Number of attention heads.
            num_groups (int): Number of groups for GroupNorm.
            num_features (int): Number of features per attention head.
            eps (float): A small constant added to the denominator for numerical stability.
        """
        super().__init__()
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_features = num_features
        self.eps = eps
        
        # Initialize GroupNorm layer
        self.group_norm = nn.GroupNorm(num_groups, num_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Headwise Group Normalization layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_heads, N, C // num_heads].
        
        Returns:
            torch.Tensor: Output tensor after applying Headwise Group Norm.
        """   
        batch_size, num_heads, N, features_per_head = x.size()
        assert features_per_head == self.num_features, "Number of features per head mismatched"
        
        x = x.view(batch_size * num_heads * N, features_per_head)
        # x = self.group_norm(x)
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        x = (x - mean) / torch.sqrt(std + self.eps)
        x = x.view(batch_size, num_heads, N, features_per_head)
        
        return x


# num_heads = 8
# num_groups = 4
# num_features = 64
# input_tensor = torch.randn(4, num_heads, num_features)  # 假设输入张量形状为 [batch_size, num_heads, num_features]
# group_norm_layer = HeadwiseGroupNorm(num_heads, num_groups, num_features)
# output = group_norm_layer(input_tensor)
# print(output.shape)  # 输出形状应该与输入形状相同 [batch_size, num_heads, num_features]

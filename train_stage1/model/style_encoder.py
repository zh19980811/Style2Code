import torch
import torch.nn as nn
import torch.nn.functional as F

class StyleEncoder(nn.Module):
    def __init__(self, input_dim=34, hidden_dims=[128, 512, 768], output_dim=1024, dropout=0.1):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.norm1 = nn.LayerNorm(hidden_dims[0])

        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.norm2 = nn.LayerNorm(hidden_dims[1])

        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.norm3 = nn.LayerNorm(hidden_dims[2])
        
        self.res_proj = nn.Linear(hidden_dims[1], hidden_dims[2]) 
        self.fc4 = nn.Linear(hidden_dims[2], output_dim)
        self.norm4 = nn.LayerNorm(output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, 34] — 原始风格向量

        returns: [B, 1024] — 风格嵌入
        """
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
    
        x = self.fc2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
    
        # ✅ 残差连接从这里开始
        residual = self.res_proj(x)
        x = self.fc3(x)
        x = self.norm3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = x + residual  # ✅ 加上 residual
    
        x = self.fc4(x)
        x = self.norm4(x)
    
        return x


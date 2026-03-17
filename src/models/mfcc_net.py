import torch
import torch.nn.functional as F
from torch import nn
import torchaudio

class AttentiveStatPool(nn.Module):
    """
    简单注意力池化：对时间维做加权平均
    输入: (B, T, H) -> 输出: (B, H)
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def _normalize_mask(self, mask: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """
        将任意形状的 mask 变成与 att_logits 同形状的 (B, T) 0/1 张量。
        target_shape: (B, T)
        """
        B, T = target_shape
        m = mask

        m = m.to(dtype=torch.float32, device=mask.device)

        if m.dim() == 1:
            if m.numel() == T:
                m = m.unsqueeze(0).expand(B, T)
            elif m.numel() == B:
                m = m.unsqueeze(1).expand(B, T)
            else:
                return torch.ones(B, T, device=mask.device, dtype=torch.float32)

        elif m.dim() == 2:
            if m.shape == (B, T):
                pass
            elif m.shape == (T, B):
                m = m.t()
            elif m.shape[0] == B:
                m = F.interpolate(m.unsqueeze(1), size=T, mode="nearest").squeeze(1)
            elif m.shape[1] == B:
                m = m.t()
                m = F.interpolate(m.unsqueeze(1), size=T, mode="nearest").squeeze(1)
            else:
                return torch.ones(B, T, device=mask.device, dtype=torch.float32)

        elif m.dim() == 3:
            if m.shape[0] != B:
                if m.shape[1] == B:
                    m = m.swapaxes(0, 1)
                else:
                    return torch.ones(B, T, device=mask.device, dtype=torch.float32)
            m = F.interpolate(m[:, :1, :], size=T, mode="nearest").squeeze(1)
        else:
            return torch.ones(B, T, device=mask.device, dtype=torch.float32)

        m = (m > 0.5).to(torch.float32)
        return m

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        B, T, _ = x.shape
        att_logits = self.att(x).squeeze(-1)  # (B, T)

        if mask is not None:
            mask = self._normalize_mask(mask, (B, T))
            att_logits = att_logits.masked_fill(mask <= 0, float("-inf"))

        att_w = torch.softmax(att_logits, dim=-1).unsqueeze(-1)  # (B, T, 1)
        wsum = att_w.sum(dim=1, keepdim=True)
        att_w = torch.where(wsum.isfinite() & (wsum > 0), att_w, torch.full_like(att_w, 1.0 / T))
        pooled = (x * att_w).sum(dim=1)  # (B, H)
        return pooled


class MFCCClassifier(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.4,
        head_init: str = "xavier_uniform",  
        head_init_std: float = 0.02,
        head_bias: float = 0.0,
    ):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": 400,
                "win_length": 400,
                "hop_length": 160,
                "n_mels": 64,
                "center": True,
                "window_fn": torch.hann_window,
            },
        )

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        encoder_out_dim = hidden_size * (2 if bidirectional else 1)

        self.encoder = nn.LSTM(
            input_size=n_mfcc,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.pool = AttentiveStatPool(encoder_out_dim)
        self.head = nn.Sequential(
            nn.Linear(encoder_out_dim, encoder_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_out_dim, encoder_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_out_dim // 2, 2),  # 二分类
        )

        self._head_init_method = head_init.lower().strip()
        self._head_init_std = float(head_init_std)
        self._head_bias = float(head_bias)
        self._init_head(self.head)

    def _init_linear(self, layer: nn.Linear):
        m = self._head_init_method
        if m == "xavier_uniform":
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("relu"))
        elif m == "xavier_normal":
            nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain("relu"))
        elif m == "kaiming_uniform":
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        elif m == "kaiming_normal":
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        elif m == "orthogonal":
            nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("relu"))
        elif m == "normal":
            nn.init.normal_(layer.weight, mean=0.0, std=self._head_init_std)
        elif m == "trunc_normal":
            nn.init.trunc_normal_(layer.weight, mean=0.0, std=self._head_init_std)
        elif m == "default":
            pass
        else:
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("relu"))

        if layer.bias is not None:
            nn.init.constant_(layer.bias, self._head_bias)

    def _init_head(self, module: nn.Module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                self._init_linear(m)

    def forward(self, input_values, attention_mask=None, labels=None):
        input_values = input_values.float().contiguous()

        mfcc_feat = self.mfcc(input_values)          
        mfcc_feat = mfcc_feat.float().contiguous()
        B, C, T_f = mfcc_feat.shape

        mask_ds = None
        if attention_mask is not None:
            am = attention_mask.to(
                device=mfcc_feat.device, dtype=torch.float32
            ).contiguous()                          
            mask_ds = F.interpolate(
                am.unsqueeze(1), size=T_f, mode="nearest"
            ).squeeze(1).contiguous()               
            
        x = mfcc_feat.transpose(1, 2).contiguous()
        x, _ = self.encoder(x)                      
        pooled = self.pool(x, mask=mask_ds)         
        logits = self.head(pooled)                  
        return logits


class MFCCRegressor(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.4,
        head_init: str = "xavier_uniform",
        head_init_std: float = 0.02,
        head_bias: float = 0.0,
    ):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": 400,
                "win_length": 400,
                "hop_length": 160,
                "n_mels": 64,
                "center": True,
                "window_fn": torch.hann_window,
            },
        )

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        encoder_out_dim = hidden_size * (2 if bidirectional else 1)

        self.encoder = nn.LSTM(
            input_size=n_mfcc,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.pool = AttentiveStatPool(encoder_out_dim)
        self.head = nn.Sequential(
            nn.Linear(encoder_out_dim, encoder_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_out_dim, encoder_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_out_dim // 2, 1), # 输出维度为1，用于回归
        )

        self._head_init_method = head_init.lower().strip()
        self._head_init_std = float(head_init_std)
        self._head_bias = float(head_bias)
        self._init_head(self.head)

    def _init_linear(self, layer: nn.Linear):
        m = self._head_init_method
        if m == "xavier_uniform":
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("relu"))
        elif m == "xavier_normal":
            nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain("relu"))
        elif m == "kaiming_uniform":
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        elif m == "kaiming_normal":
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        elif m == "orthogonal":
            nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("relu"))
        elif m == "normal":
            nn.init.normal_(layer.weight, mean=0.0, std=self._head_init_std)
        elif m == "trunc_normal":
            nn.init.trunc_normal_(layer.weight, mean=0.0, std=self._head_init_std)
        elif m == "default":
            pass
        else:
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("relu"))

        if layer.bias is not None:
            nn.init.constant_(layer.bias, self._head_bias)

    def _init_head(self, module: nn.Module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                self._init_linear(m)

    def forward(self, input_values, attention_mask=None, labels=None):
        input_values = input_values.float().contiguous()

        mfcc_feat = self.mfcc(input_values)
        mfcc_feat = mfcc_feat.float().contiguous()
        B, C, T_f = mfcc_feat.shape

        mask_ds = None
        if attention_mask is not None:
            am = attention_mask.to(
                device=mfcc_feat.device, dtype=torch.float32
            ).contiguous()
            mask_ds = F.interpolate(
                am.unsqueeze(1), size=T_f, mode="nearest"
            ).squeeze(1).contiguous()
            
        x = mfcc_feat.transpose(1, 2).contiguous()
        x, _ = self.encoder(x)
        pooled = self.pool(x, mask=mask_ds)
        
        preds = self.head(pooled).squeeze(-1)
        
        loss = None
        if labels is not None:
            labels = labels.to(preds.dtype).view(-1)
            loss_fct = nn.MSELoss()
            loss = loss_fct(preds, labels)
            
        return {"loss": loss, "preds": preds}
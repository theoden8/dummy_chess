"""dc0 neural network: SE-ResNet with policy + WDL value heads.

Architecture follows LC0/AlphaZero with Squeeze-Excitation residual blocks.

Input:  (batch, 22, 8, 8) float tensor
Output: policy logits (batch, 4672), WDL logits (batch, 3)
"""

import torch
import torch.nn

import move_encoding


# --- Input planes ---
# 12 piece planes (6 types x 2 colors)
# 2 repetition planes (1x, 2x)
# 1 color-to-move plane
# 1 fullmove count plane
# 4 castling rights planes (K, Q, k, q)
# 1 en passant plane
# 1 halfmove clock plane
NUM_INPUT_PLANES = 22


class SEBlock(torch.nn.Module):
    """Squeeze-Excitation block."""

    def __init__(self, channels: int, se_ratio: int = 4):
        super().__init__()
        mid = channels // se_ratio
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Linear(channels, mid)
        self.fc2 = torch.nn.Linear(mid, 2 * channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)
        s = torch.relu(self.fc1(s))
        s = self.fc2(s)
        # Split into scale (sigmoid) and bias
        w, b_bias = s.view(b, 2, c, 1, 1).chunk(2, dim=1)
        w = w.squeeze(1)
        b_bias = b_bias.squeeze(1)
        return torch.sigmoid(w) * x + b_bias


class ResBlock(torch.nn.Module):
    """SE-Residual block: conv -> BN -> ReLU -> conv -> BN -> SE -> add -> ReLU."""

    def __init__(self, channels: int, se_ratio: int = 4):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(channels)
        self.conv2 = torch.nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, se_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return torch.relu(out + residual)


class DC0Network(torch.nn.Module):
    """SE-ResNet with policy and WDL value heads.

    Args:
        n_blocks: number of residual blocks (default 6)
        n_filters: number of convolutional filters (default 128)
        se_ratio: squeeze-excitation reduction ratio (default 4)
        policy_channels: intermediate channels in policy head (default 32)
        value_channels: intermediate channels in value head (default 32)
        value_fc_size: hidden size in value head FC layer (default 128)
    """

    def __init__(
        self,
        n_blocks: int = 6,
        n_filters: int = 128,
        se_ratio: int = 4,
        policy_channels: int = 32,
        value_channels: int = 32,
        value_fc_size: int = 128,
    ):
        super().__init__()

        # Store config for serialization
        self.n_blocks = n_blocks
        self.n_filters = n_filters
        self.se_ratio = se_ratio
        self.policy_channels = policy_channels
        self.value_channels = value_channels
        self.value_fc_size = value_fc_size

        # Input conv
        self.input_conv = torch.nn.Conv2d(
            NUM_INPUT_PLANES, n_filters, 3, padding=1, bias=False
        )
        self.input_bn = torch.nn.BatchNorm2d(n_filters)

        # Residual tower
        self.blocks = torch.nn.ModuleList(
            [ResBlock(n_filters, se_ratio) for _ in range(n_blocks)]
        )

        # Policy head
        self.policy_conv1 = torch.nn.Conv2d(n_filters, policy_channels, 1, bias=False)
        self.policy_bn = torch.nn.BatchNorm2d(policy_channels)
        self.policy_conv2 = torch.nn.Conv2d(
            policy_channels, move_encoding.NUM_MOVE_TYPES, 1
        )

        # Value head
        self.value_conv = torch.nn.Conv2d(n_filters, value_channels, 1, bias=False)
        self.value_bn = torch.nn.BatchNorm2d(value_channels)
        self.value_fc1 = torch.nn.Linear(value_channels * 64, value_fc_size)
        self.value_fc2 = torch.nn.Linear(value_fc_size, 3)  # WDL

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: input tensor of shape (batch, 22, 8, 8)

        Returns:
            policy_logits: (batch, 4672) raw logits, NOT masked or softmaxed
            wdl_logits: (batch, 3) win/draw/loss logits, NOT softmaxed
        """
        # Input block
        out = torch.relu(self.input_bn(self.input_conv(x)))

        # Residual tower
        for block in self.blocks:
            out = block(out)

        # Policy head: conv -> BN -> ReLU -> conv -> flatten
        p = torch.relu(self.policy_bn(self.policy_conv1(out)))
        p = self.policy_conv2(p)  # (batch, 73, 8, 8)
        # Reshape to (batch, 8, 8, 73) then flatten to (batch, 4672)
        # Policy index = from_sq * 73 + move_type
        # from_sq layout: file + rank*8, tensor layout: [move_type, rank, file]
        # We need: for each square (file, rank), 73 move types
        # p shape is (batch, 73, 8, 8) where dim2=rank, dim3=file
        # Permute to (batch, 8, 8, 73) = (batch, rank, file, move_type)
        # Then reshape so index = (rank*8 + file)*73 + move_type = from_sq*73 + mt
        p = p.permute(0, 2, 3, 1).contiguous()  # (batch, rank, file, 73)
        p = p.view(p.shape[0], move_encoding.POLICY_SIZE)  # (batch, 4672)

        # Value head: conv -> BN -> ReLU -> flatten -> FC -> ReLU -> FC
        v = torch.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.shape[0], -1)  # (batch, value_channels * 64)
        v = torch.relu(self.value_fc1(v))
        v = self.value_fc2(v)  # (batch, 3)

        return p, v

    def predict(
        self, x: torch.Tensor, legal_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with legal move masking and softmax.

        Args:
            x: input tensor (batch, 22, 8, 8)
            legal_mask: boolean tensor (batch, 4672), True for legal moves

        Returns:
            policy: (batch, 4672) probability distribution over moves
            wdl: (batch, 3) win/draw/loss probabilities
        """
        policy_logits, wdl_logits = self.forward(x)

        # Mask illegal moves with -inf before softmax
        policy_logits = policy_logits.masked_fill(~legal_mask, float("-inf"))
        policy = torch.softmax(policy_logits, dim=1)

        wdl = torch.softmax(wdl_logits, dim=1)

        return policy, wdl

    def value_from_wdl(self, wdl: torch.Tensor) -> torch.Tensor:
        """Convert WDL probabilities to scalar value in [-1, 1].

        Args:
            wdl: (batch, 3) probabilities [P(win), P(draw), P(loss)]

        Returns:
            (batch,) scalar value = P(win) - P(loss)
        """
        return wdl[:, 0] - wdl[:, 2]

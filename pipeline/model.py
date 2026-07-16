"""
Vector Equities MTNN Model — port of vector-hoops train_mtnn.py architectures
ResidualTower, GatedFusion, ConcatFusion, TransformerFusion, SkillTowers, MTNN

Solo personal project, no connection to employer, built with public/free-tier only
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class _ResBlock(nn.Module):
    def __init__(self, d: int, d_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d, d_hidden)
        self.ln1 = nn.LayerNorm(d_hidden)
        self.fc2 = nn.Linear(d_hidden, d)
        self.ln2 = nn.LayerNorm(d)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.ln2(self.fc2(F.gelu(self.ln1(self.fc1(y)))) + y)

class ResidualTower(nn.Module):
    def __init__(self, d_in: int, d_out: int=24, d_hidden: int=96, n_blocks: int=1):
        super().__init__()
        d_cat = d_in*2
        self.fc1 = nn.Linear(d_cat, d_hidden)
        self.ln1 = nn.LayerNorm(d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.ln2 = nn.LayerNorm(d_out)
        self.skip = nn.Linear(d_cat, d_out) if d_cat!=d_out else nn.Identity()
        self.blocks = nn.ModuleList([_ResBlock(d_out, d_hidden) for _ in range(max(0, n_blocks-1))])

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x*m, m], dim=-1)
        y = self.ln2(self.fc2(F.gelu(self.ln1(self.fc1(h)))) + self.skip(h))
        for blk in self.blocks:
            y = blk(y)
        return y

class GatedFusion(nn.Module):
    def __init__(self, n_towers: int, d_tower: int, n_seasons: int, d_season: int=12, d_emb: int=48, d_hidden: int=192):
        super().__init__()
        self.season_emb = nn.Embedding(n_seasons, d_season)
        self.gate = nn.Linear(d_tower, 1)
        self.attn = nn.Sequential(nn.Linear(d_tower, d_tower), nn.Tanh(), nn.Linear(d_tower, 1))
        self.fuse = nn.Sequential(nn.Linear(d_tower + d_season, d_hidden), nn.GELU(), nn.LayerNorm(d_hidden), nn.Linear(d_hidden, d_emb))

    def forward(self, tower_stack: torch.Tensor, season_ids: torch.Tensor) -> torch.Tensor:
        scores = self.attn(tower_stack).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        gates = torch.sigmoid(self.gate(tower_stack).squeeze(-1))
        mixed = (tower_stack * weights.unsqueeze(-1) * gates.unsqueeze(-1)).sum(1)
        s = self.season_emb(season_ids)
        return F.normalize(self.fuse(torch.cat([mixed, s], dim=-1)), dim=-1)

class ConcatFusion(nn.Module):
    def __init__(self, n_towers: int, d_tower: int, n_seasons: int, d_season: int=12, d_emb: int=48, d_hidden: int=256):
        super().__init__()
        self.season_emb = nn.Embedding(n_seasons, d_season)
        d_in = n_towers*d_tower + d_season
        self.fuse = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.LayerNorm(d_hidden), nn.Linear(d_hidden, d_emb))

    def forward(self, tower_stack: torch.Tensor, season_ids: torch.Tensor) -> torch.Tensor:
        flat = tower_stack.reshape(tower_stack.size(0), -1)
        s = self.season_emb(season_ids)
        return F.normalize(self.fuse(torch.cat([flat, s], dim=-1)), dim=-1)

class TransformerFusion(nn.Module):
    def __init__(self, n_towers: int, d_tower: int, n_seasons: int, d_season: int=12, d_emb: int=48, d_model: int=96, n_layers: int=4, n_heads: int=4, ff: int=256, dropout: float=0.1):
        super().__init__()
        self.tower_proj = nn.Linear(d_tower, d_model)
        self.season_emb = nn.Embedding(n_seasons, d_season)
        self.season_proj = nn.Linear(d_season, d_model)
        self.cls = nn.Parameter(torch.randn(1,1,d_model)*0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ff, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, d_emb)

    def forward(self, tower_stack: torch.Tensor, season_ids: torch.Tensor) -> torch.Tensor:
        b = tower_stack.size(0)
        tok = self.tower_proj(tower_stack)
        s = self.season_proj(self.season_emb(season_ids)).unsqueeze(1)
        cls = self.cls.expand(b,-1,-1)
        x = self.encoder(torch.cat([cls, s, tok], dim=1))
        return F.normalize(self.out(x[:,0]), dim=-1)

class SkillTowers(nn.Module):
    def __init__(self, d_emb: int, n_skills: int, d_hidden: int=16):
        super().__init__()
        self.towers = nn.ModuleList([nn.Sequential(nn.Linear(d_emb, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1)) for _ in range(n_skills)])

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return torch.cat([t(emb) for t in self.towers], dim=-1)

class EquitiesMTNN(nn.Module):
    def __init__(self, fam_dims: dict, n_seasons: int, d_tower: int=24, d_tower_hidden: int=96, d_emb: int=48,
                 n_game: int=14, n_skills: int=12, d_skill_hidden: int=16,
                 n_sectors: int=11, n_archetypes: int=8,
                 fusion_mode: str="gated", n_tower_blocks: int=1, mlp_heads: bool=False, d_head_hidden: int=64,
                 d_model: int=96, n_fusion_layers: int=4, n_attn_heads: int=4, d_fusion_hidden: int=None):
        super().__init__()
        self.families = sorted(fam_dims)
        self.fusion_mode = fusion_mode
        self.towers = nn.ModuleDict({fam: ResidualTower(fam_dims[fam], d_out=d_tower, d_hidden=d_tower_hidden, n_blocks=n_tower_blocks) for fam in self.families})

        if fusion_mode=="concat":
            self.fusion = ConcatFusion(len(self.families), d_tower, n_seasons, d_emb=d_emb, **({} if d_fusion_hidden is None else {"d_hidden":d_fusion_hidden}))
        elif fusion_mode=="transformer":
            self.fusion = TransformerFusion(len(self.families), d_tower, n_seasons, d_emb=d_emb, d_model=d_model, n_layers=n_fusion_layers, n_heads=n_attn_heads, **({} if d_fusion_hidden is None else {"ff":d_fusion_hidden}))
        else:
            self.fusion = GatedFusion(len(self.families), d_tower, n_seasons, d_emb=d_emb, **({} if d_fusion_hidden is None else {"d_hidden":d_fusion_hidden}))

        def head(k): 
            return nn.Sequential(nn.Linear(d_emb, d_head_hidden), nn.GELU(), nn.Linear(d_head_hidden, k)) if mlp_heads else nn.Linear(d_emb, k)

        self.archetype_head = head(n_archetypes)
        self.sector_head = head(n_sectors)
        self.profile_head = head(n_game)
        self.next_profile_head = head(n_game)
        self.valuation_head = nn.Linear(d_emb, 1)  # EV/EBITDA z
        self.market_head = nn.Linear(d_emb, 1)  # next excess ret
        self.vol_head = nn.Linear(d_emb, 1)  # volatility
        self.health_head = nn.Linear(d_emb, 1)  # Altman Z
        self.payout_head = nn.Linear(d_emb, 1)  # Div growth
        self.mgmt_head = nn.Linear(d_emb, 1)  # Comp efficiency
        self.own_head = nn.Linear(d_emb, 1)
        self.skill_towers = SkillTowers(d_emb, n_skills, d_hidden=d_skill_hidden) if n_skills else None

    def encode(self, xs, ms, season_ids):
        parts = torch.stack([self.towers[fam](xs[fam], ms[fam]) for fam in self.families], dim=1)
        return self.fusion(parts, season_ids)

    def forward(self, xs, ms, season_ids):
        emb = self.encode(xs, ms, season_ids)
        out = {
            "archetype": self.archetype_head(emb),
            "sector": self.sector_head(emb),
            "profile": self.profile_head(emb),
            "next_profile": self.next_profile_head(emb),
            "valuation": self.valuation_head(emb).squeeze(-1),
            "market": self.market_head(emb).squeeze(-1),
            "vol": self.vol_head(emb).squeeze(-1),
            "health": self.health_head(emb).squeeze(-1),
            "payout": self.payout_head(emb).squeeze(-1),
            "mgmt": self.mgmt_head(emb).squeeze(-1),
            "own": self.own_head(emb).squeeze(-1),
        }
        if self.skill_towers is not None:
            out["skills"] = self.skill_towers(emb)
        return emb, out

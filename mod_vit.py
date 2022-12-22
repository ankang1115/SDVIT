import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

device = torch.device('cuda:0')
# reformer = torch.load('../reformer/AutoEncoder.pkl', map_location='cpu').to(device)

class DCP(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxp = torch.nn.MaxPool2d(3, 1, 1)
        self.avgp = torch.nn.AvgPool2d(3, 1, 1)
        self._1conv1 = torch.nn.Conv2d(12, 3, 1)
        # self.post = torch.nn.AvgPool2d(6, 3, 3)
        self.s = torch.nn.AdaptiveAvgPool2d(1)

        
    def forward(self, x, **kwargs):
        x_delta = torch.zeros(x.shape)
        x_delta[:,:, :, 1:] = x[:, :, :, 1:] - x[:, :, :, :-1]

        y_delta = torch.zeros(x.shape)
        y_delta[:, :, 1:, :] = x[:, :, 1:, :] - x[:, :, : -1, :]
        # y_delta = torch.zeros(x.shape[-2:])
        # y_delta[1:, :] = x[1:, :] - x[:-1, :]
        x1 = self.maxp(x)
        x2 = self.avgp(x)
        xs = self._1conv1(torch.concat([x1, x2, x_delta, y_delta], dim = 1))
        s_ = self.s(xs)
        se_ = xs * s_

        return se_ + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for ind, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x
            x = ff(x) + x
            if ind == 2:
                x_mid = x
        return x, x_mid

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.dcp = DCP()
        self.detail_process1 = torch.nn.Conv2d(3, 1, 3) 
        self.detail_process2 = torch.nn.MaxPool2d(16,4)
        self.detail_process3 = torch.nn.Linear(2704, 1024)

    def forward(self, img):

        x_detail = self.dcp(img)
        x_detail = self.detail_process1(x_detail)
        x_detail = self.detail_process2(x_detail)
        x_detail = x_detail.view(x_detail.shape[0], -1)
        x_detail = self.detail_process3(x_detail)

        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x, x_mid = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x + x_detail)
        return self.mlp_head(x), x_mid

model = ViT(
    channels = 3,
    image_size = 224,
    patch_size = 32,
    num_classes = 2,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.rand(1,3,224,224)
model(img)



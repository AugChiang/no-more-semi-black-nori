import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        residual = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)

        y = self.conv1(x)
        y = self.conv2(y)
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.conv3(y)

        y = self.dropout1(y)
        x = residual + y * self.beta

        residual = x
        y = x.permute(0, 2, 3, 1)
        y = self.norm2(y)
        y = y.permute(0, 3, 1, 2)

        y = self.conv4(y)
        y = self.sg(y)
        y = self.conv5(y)

        y = self.dropout2(y)
        x = residual + y * self.gamma

        return x

class FFTBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(c * 2, c * 2, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 2, 1, 1, 0)
        )
        nn.init.zeros_(self.main[-1].weight)
        nn.init.zeros_(self.main[-1].bias)

    def forward(self, x):
        batch, c, h, w = x.shape
        
        # FFT to frequency domain
        ffted = torch.fft.rfft2(x, norm='ortho')
        # Separate real and imaginary parts
        real = ffted.real
        imag = ffted.imag
        
        # Concatenate real and imaginary parts along channel dimension
        freq = torch.cat([real, imag], dim=1)
        
        # Learnable spectral filtering
        freq = self.main(freq)
        
        # Restore complex representation
        real, imag = freq.chunk(2, dim=1)
        ffted = torch.complex(real, imag)
        
        # Inverse FFT back to spatial domain
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')
        
        return output

class DualDomainNAFNet(nn.Module):
    def __init__(
        self,
        img_channel=1,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=(1, 1, 1, 1),
        dec_blk_nums=(1, 1, 1, 1),
    ):
        super().__init__()
        self.img_channel = img_channel
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks.append(
            nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        )
        
        # FFT Block in the middle
        self.fft_block = FFTBlock(chan)

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )

        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks[0](x)
        # Apply FFT refinement in the bottleneck
        x = x + self.fft_block(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

if __name__ == "__main__":
    # Test model
    model = DualDomainNAFNet(width=16)
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

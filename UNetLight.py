import torch
import torch.nn as nn

class UNetLight(nn.Module):
    def __init__(self):
        super(UNetLight, self).__init__()
        
        # Contracting Path
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.encoder1 = self.conv_block(3, 16)
        self.encoder2 = self.conv_block(16, 32)
        self.encoder3 = self.conv_block(32, 64)
        self.encoder4 = self.conv_block(64, 128)
        
        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)
        
        # Expansive Path
        self.decoder4 = self.deconv_block(256, 128)
        self.up4 = self.up_block(256,128)
        self.decoder3 = self.deconv_block(128, 64)
        self.up3 = self.up_block(128,64)
        self.decoder2 = self.deconv_block(64, 32)
        self.up2 = self.up_block(64,32)
        self.decoder1 = self.deconv_block(32, 16)
        self.up1 = self.up_block(32,16)
        
        # Output layer
        self.output_layer = nn.Sequential(
          nn.Conv2d(16, 1, kernel_size=1),
          nn.Sigmoid()
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding ="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def up_block(self,in_channels,out_channels):
      return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
      )


    def forward(self, x):
        # Contracting Path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Expansive Path
        dec4 = self.decoder4(torch.cat([enc4, self.up4(bottleneck)], dim=1))
        dec3 = self.decoder3(torch.cat([enc3, self.up3(dec4)], dim=1))
        dec2 = self.decoder2(torch.cat([enc2, self.up2(dec3)], dim=1))
        dec1 = self.decoder1(torch.cat([enc1, self.up1(dec2)], dim=1))
        
        # Output layer
        output = self.output_layer(dec1)
        
        return output
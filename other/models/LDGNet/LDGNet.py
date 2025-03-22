
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import swinplus
import ptflops
# import clip
# model
def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3, stride=1,padding=1,bias=False),
        nn.BatchNorm3d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer

class residual_block(nn.Module):

    def __init__(self, in_channel,out_channel):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel,out_channel)
        self.conv2 = conv3x3x3(out_channel,out_channel)
        self.conv3 = conv3x3x3(out_channel,out_channel)

    def forward(self, x): #(1,1,100,9,9)
        x1 = F.relu(self.conv1(x), inplace=True)
        x2 = F.relu(self.conv2(x1), inplace=True)
        x3 = self.conv3(x2)

        out = F.relu(x1+x3, inplace=True)
        return out

class D_Res_3d_CNN(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2, CLASS_NUM, patch_size, n_bands, embed_dim):
        super(D_Res_3d_CNN, self).__init__()
        self.n_bands = n_bands
        self.block1 = residual_block(in_channel,out_channel1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1,2,2),padding=(0,1,1),stride=(4,2,2))
        self.block2 = residual_block(out_channel1,out_channel2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(1,2,2),stride=(1,2,2), padding=(0,1,1))
        self.conv1 = nn.Conv3d(in_channels=out_channel2,out_channels=32,kernel_size=(1,3,3), bias=False)
        self.patch_size = patch_size
        # self.final_feat_dim = 128
        self.fc = nn.Linear(in_features=32, out_features=embed_dim, bias=False)
        self.classifier = nn.Linear(in_features=32, out_features=CLASS_NUM, bias=False)
        self.up = nn.Upsample(scale_factor=4.15, mode='bilinear', align_corners=True)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
    def _get_layer_size(self):
        with torch.no_grad():
            x = torch.zeros((1,1, self.n_bands,
                             self.patch_size, self.patch_size))
            x = self.block1(x)
            x = self.maxpool1(x)
            x = self.block2(x)
            x = self.maxpool2(x)
            x = self.conv1(x)
            x = x.view(x.shape[0],-1)
            s = x.size()[1]
        return s

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.maxpool1(x)
        x = self.block2(x)
        x = self.maxpool2(x)
        x = self.conv1(x)


        x1 = x.squeeze(2)
        x1 = self.up(x1)
        x1 = x1.permute(0, 2, 3, 1)
        y = self.classifier(x1)
        y = y.permute(0, 3, 1, 2)

        x = self.global_avg_pool(x)

        x = x.view(x.size(0), -1)
        proj = self.fc(x)

        return y, proj


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class LDGnet(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 inchannel,
                 vision_patch_size: int,
                 num_classes,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length


        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.numclass = num_classes
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # for p in self.parameters():
        #     p.requires_grad = False

        # self.visual = ConvNet_HSI_MI(inchannel=inchannel, outchannel=embed_dim, num_classes=num_classes, patch_size=vision_patch_size)
        self.visual = swinplus.CustomSwinTransformer()
        self.visual1 = D_Res_3d_CNN(1,8,16,num_classes, vision_patch_size, inchannel, embed_dim)
        self.initialize_parameters()

        self.classifier = nn.Sequential(
            nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(),
            nn.Conv2d(num_classes, num_classes, kernel_size=1, stride=1, padding=0)
        )
        self.classifier2 = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(),
            nn.Conv2d(num_classes, num_classes, kernel_size=1, stride=1, padding=0)
        )
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, mode):
        # return self.visual(image.type(self.dtype), mode)
        return self.visual(image)

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def predict(self,image):
        imgage_prob1 = self.encode_image(image, mode='train')
        imgage_prob = self.classifier(imgage_prob1)
        return imgage_prob

    def forward(self, image, text_queue_1=None, text_queue_2=None):
        # imgage_prob, image_features = self.encode_image(image, mode='train')
        imgage_prob1 = self.encode_image(image, mode='train')
        imgage_prob = self.classifier(imgage_prob1)

        # if self.training:
        #     text_features = self.encode_text(text)
        # a = imgage_prob
        # new_a = torch.zeros_like(a)  # 创建一个与 a 形状相同的新张量来存储结果
        # for i in range(self.numclass):
        #     for j in range(16):
        #         new_a[j][i] = text_features[i] * a[j][i]
        #
        # a = self.classifier2(new_a)

        #     # normalized features
        #     image_features = image_features / image_features.norm(dim=1, keepdim=True)
        #     text_features = text_features / text_features.norm(dim=1, keepdim=True)
        #
        #     # cosine similarity as logits
        #     logit_scale = self.logit_scale.exp()
        #     logits_per_image = logit_scale * image_features @ text_features.t()
        #     logits_per_text = logit_scale * text_features @ image_features.t()
        #
        #     loss_img = F.cross_entropy(logits_per_image, label.long())
        #
        #     loss_text = F.cross_entropy(logits_per_text, label.long())
        #     loss_clip = (loss_img + loss_text)/2
        #     # q1
        #     # normalized features
        #     text_features_q1 = text_features_q1 / text_features_q1.norm(dim=1, keepdim=True)
        #
        #     # cosine similarity as logits
        #     logit_scale = self.logit_scale.exp()
        #     logits_per_image = logit_scale * image_features @ text_features_q1.t()
        #     logits_per_text = logit_scale * text_features_q1 @ image_features.t()
        #
        #     loss_img = F.cross_entropy(logits_per_image, label.long())
        #     loss_text = F.cross_entropy(logits_per_text, label.long())
        #     loss_q1 = (loss_img + loss_text)/2
        #     # q2
        #     # normalized features
        #     text_features_q2 = text_features_q2 / text_features_q2.norm(dim=1, keepdim=True)
        #
        #     # cosine similarity as logits
        #     logit_scale = self.logit_scale.exp()
        #     logits_per_image = logit_scale * image_features @ text_features_q2.t()
        #     logits_per_text = logit_scale * text_features_q2 @ image_features.t()
        #
        #     loss_img = F.cross_entropy(logits_per_image, label.long())
        #     loss_text = F.cross_entropy(logits_per_text, label.long())
        #     loss_q2 = (loss_img + loss_text)/2
        #     return loss_clip, (loss_q1+loss_q2)/2, imgage_prob
        # else:
        #     return torch.tensor(0).long(), imgage_prob
        return imgage_prob
if __name__ == '__main__':
    class_name = ['Seagrass bed',
                  'Spartina alterniflora',
                  'Reed',
                  'Tamarix',
                  'Tidal flat',
                  'Sparse vegetation',
                  'Pond',
                  'Yellow River',
                  'Sea',
                  'Cloud']
    queue = {
        'Seagrass bed': [
            'The Seagrass bed is next to the Sea',
            'The Seagrass bed is green'
        ],
        'Spartina alterniflora': [
            'The Spartina alterniflora is next to the Sea',
            'The Spartina alterniflora is light green'
        ],
        'Reed': [
            'The Reed is next to the Pond',
            'The Reed is dark green'
        ],
        'Tamarix': [
            'The Tamarix is next to the Reed',
            'The Tamarix appears golden-orange'
        ],
        'Tidal flat': [
            'Tidal flat is next to Pond',
            'Tidal flat appears tan'
        ],
        'Sparse vegetation': [
            'Sparse vegetation is next to Tidal flat',
            'Sparse vegetation appears light green'
        ],
        'Pond': [
            'Pond is next to Tidal flat',
            'Pond is blue'  # 注意这里修正了多余的引号
        ],
        'Yellow River': [
            'Yellow River is next to Sea',
            'Yellow River is brown'
        ],
        'Sea': [
            'Sea is next to Spartina alterniflora',
            'Sea is blue'
        ],
        'Cloud': [
            'Cloud is next to Spartina alterniflora',
            'Cloud is white'
        ]
    }




    x = torch.randn(16,4,128,128)
    label = torch.randn(16,128,128)
    pretrained_dict = torch.load('./ViT-B-32.pt', map_location="cpu").state_dict()
    embed_dim = pretrained_dict["text_projection"].shape[1]
    context_length = pretrained_dict["positional_embedding"].shape[0]
    vocab_size = pretrained_dict["token_embedding.weight"].shape[0]
    transformer_width = pretrained_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = 3
    model = LDGnet(embed_dim=128,inchannel=4,vision_patch_size=13,num_classes=10,
                   context_length=context_length, vocab_size=vocab_size,transformer_width= transformer_width, transformer_heads=transformer_heads, transformer_layers=transformer_layers)
    flops, params = ptflops.get_model_complexity_info(model, (4, 128, 128), as_strings=True,
                                                      print_per_layer_stat=True, verbose=True)
    print('FLOPs:  ' + flops)
    print('Params: ' + params)
    # text = torch.cat(
    #     [clip.tokenize(f'A hyperspectral image of {class_name[k]}') for k in range(10)])
    # text_queue_1 = [queue[class_name[k]][0] for k in range(10)]
    # text_queue_2 = [queue[class_name[k]][1] for k in range(10)]
    # text_queue_1 = torch.cat([clip.tokenize(k).to(text.device) for k in text_queue_1])
    # text_queue_2 = torch.cat([clip.tokenize(k).to(text.device) for k in text_queue_2])
    # out1,out2 = model(x,text,label,text_queue_1,text_queue_2)

    # print(out1.shape)
    # print(out2.shape)

from model.build_BiSeNet import BiSeNet
import torch
from torchscan import summary
from model.discriminator import FCDiscriminator, Light_Discriminator


num_classes = 19
context_path = 'resnet18'
model = BiSeNet(num_classes, context_path)

#discriminator model
#d_model = FCDiscriminator(num_classes=num_classes)
d_model = Light_Discriminator(num_classes=num_classes)

if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()
    d_model = torch.nn.DataParallel(d_model).cuda()


model.eval()
d_model.eval()
summary(model, (3, 1024, 512), receptive_field=True)
summary(d_model, (19,1024,512), receptive_field=True)




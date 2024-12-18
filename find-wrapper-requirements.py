import torch
from model import CustomSegmentationModel
from torch import nn
from torchvision.models.segmentation import fcn_resnet50

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
student_model = CustomSegmentationModel(num_classes=21).to(device)
teacher_model = fcn_resnet50(weights='DEFAULT').to(device)

# Wrap teacher model
class TeacherWrapper(nn.Module):
    def __init__(self, teacher_model):
        super().__init__()
        self.teacher = teacher_model
        self.layer1 = self.teacher.backbone.layer1
        self.layer2 = self.teacher.backbone.layer2
        self.layer3 = self.teacher.backbone.layer3
        self.layer4 = self.teacher.backbone.layer4

    def forward(self, x):
        x = self.teacher.backbone.conv1(x)
        x = self.teacher.backbone.bn1(x)
        x = self.teacher.backbone.relu(x)
        x = self.teacher.backbone.maxpool(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return {'features': [f1, f2, f3, f4]}

teacher_model = TeacherWrapper(teacher_model).to(device)

# Generate a random input
x = torch.randn(1, 3, 224, 224).to(device)

# Forward pass through student model
student_features = student_model(x)['features']
print("Student Feature Map Channels:")
for i, sf in enumerate(student_features):
    print(f"Layer {i}: {sf.shape}")

# Forward pass through teacher model
teacher_features = teacher_model(x)['features']
print("\nTeacher Feature Map Channels:")
for i, tf in enumerate(teacher_features):
    print(f"Layer {i}: {tf.shape}")
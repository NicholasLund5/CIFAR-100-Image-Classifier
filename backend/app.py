from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from torch import nn
import torch
from torchvision import transforms, datasets
from PIL import Image
import os

class SqueezeExcitationBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.global_avg_pool(x).view(batch_size, channels)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)
        return x * y

class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=0.3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitationBlock(out_channels, reduction)

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False) \
                        if in_channels != out_channels or stride != 1 else nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.se(x)  
        return nn.ReLU()(x + shortcut)

class CIFAR100Model(nn.Module):
    def __init__(self, input_shape: int, width_multiplier: int, num_classes: int):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        self.stage1 = self._make_stage(16, 16 * width_multiplier, num_blocks=2, stride=1)
        
        self.stage2 = self._make_stage(16 * width_multiplier, 32 * width_multiplier, num_blocks=2, stride=2)
        
        self.stage3 = self._make_stage(32 * width_multiplier, 64 * width_multiplier, num_blocks=2, stride=2)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(64 * width_multiplier, num_classes)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            layers.append(
                ResidualSEBlock(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    stride=stride if i == 0 else 1
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x) 
        x = self.fc(x)
        return x


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


@app.on_event("startup")
def load_model():
    global model, class_names, device
    model_path = "./CIFAR100_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = datasets.CIFAR100(
        root="data",
        train=True,
        download=True,  
        transform=None
    )
    class_names = train_data.classes 

    model = CIFAR100Model(input_shape=3, width_multiplier=10, num_classes=len(class_names)).to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()


transform = transforms.Compose([
    transforms.Resize((32, 32)),  
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])


@app.post("/classify/")
async def classify_image(image: UploadFile = File(...)):
    try:
        print(f"File received: {image.filename}")
        img = Image.open(image.file).convert("RGB")
        
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted_class = torch.max(outputs, 1)
            prediction = class_names[predicted_class.item()]  #
            prediction = prediction.replace("_", " ").capitalize()

        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Root Endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the CIFAR-100 Image Classifier API!"}
import torch
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import requests
import matplotlib.patches as patches
import sys

sys.path.append("/content/DA6401-Assignement-II---EE21D063")
from models.vgg11 import VGG11Encoder
from models.multitask import MultiTaskPerceptionModel

device = torch.device("cpu")
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# section 2.4 
img_path = "/content/oxford-iiit-pet/images/Abyssinian_11.jpg"
image = np.array(Image.open(img_path).convert("RGB"))
input_tensor = transform(image=image)['image'].unsqueeze(0)
encoder = VGG11Encoder(in_channels=3)
try:
    encoder.load_state_dict({
        k.replace("encoder.", ""): v
        for k, v in torch.load("checkpoints/classifier.pth", map_location='cpu')["state_dict"].items()
        if "encoder." in k
    }, strict=False)
except:
    pass
encoder.eval()
with torch.no_grad():
    _, features = encoder(input_tensor, return_features=True)
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(image)
axs[0].set_title("Original")
axs[1].imshow(features['pool1_pre'][0, 0].numpy(), cmap='viridis')
axs[1].set_title("Block 1 (Edges)")
axs[2].imshow(features['pool5_pre'][0, 0].numpy(), cmap='viridis')
axs[2].set_title("Block 5 (Semantic)")
plt.savefig("2_4_Feature_Maps.png")
print("Saved 2_4_Feature_Maps.png")
#------------------------------------------------------------------------------------------------------------
# section 2.6
model = MultiTaskPerceptionModel(
    classifier_path="checkpoints/classifier.pth",
    localizer_path="checkpoints/localizer.pth",
    unet_path="checkpoints/unet.pth"
)
model.eval()
sample_images = [
    "Abyssinian_106.jpg", "Bengal_115.jpg", "Bombay_125.jpg", "Maine_Coon_133.jpg", "Ragdoll_112.jpg"
]
fig, axs = plt.subplots(5, 2, figsize=(8, 15))
for i, name in enumerate(sample_images):
    img = np.array(Image.open(f"/content/oxford-iiit-pet/images/{name}").convert("RGB"))
    tens = transform(image=img)['image'].unsqueeze(0)
    with torch.no_grad():
        mask = torch.argmax(model(tens)['segmentation'][0], dim=0).numpy()
    axs[i, 0].imshow(A.Resize(224, 224)(image=img)['image'])
    axs[i, 0].set_title("Original")
    axs[i, 1].imshow(mask, cmap="viridis")
    axs[i, 1].set_title("Predicted Mask")
plt.tight_layout()
plt.savefig("2_6_Segmentation_Masks.png")
print("Saved 2_6_Segmentation_Masks.png")
#------------------------------------------------------------------------------------------------------------
#section 2.7 
urls = [
    "https://images.unsplash.com/photo-1518717758536-85ae29035b6d?auto=format&fit=crop&w=600&q=80",
    "https://images.unsplash.com/photo-1495360010541-f48722b34f7d?auto=format&fit=crop&w=600&q=80",
    "https://images.unsplash.com/photo-1507146426996-ef05306b995a?auto=format&fit=crop&w=600&q=80"
]
fig, axs = plt.subplots(3, 2, figsize=(10, 12))
for i, url in enumerate(urls):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        with open("temp.jpg", "wb") as f:
            f.write(response.content)
    except Exception as e:
        print(f"Skipping broken URL: {url}")
        continue
    img = np.array(Image.open("temp.jpg").convert("RGB"))
    tens = transform(image=img)['image'].unsqueeze(0)

    with torch.no_grad():
        out = model(tens)
    cx, cy, w, h = out['localization'][0].numpy()
    if max(cx, cy, w, h) <= 2.0:
        cx, cy, w, h = cx * 224.0, cy * 224.0, w * 224.0, h * 224.0
    ax_img = axs[i, 0]
    ax_img.imshow(A.Resize(224, 224)(image=img)['image'])
    ax_img.add_patch(
        patches.Rectangle(
            (cx - w/2, cy - h/2),
            w, h,
            linewidth=3,
            edgecolor='r',
            facecolor='none'
        )
    )
    ax_img.set_title("Detection")
    axs[i, 1].imshow(
        torch.argmax(out['segmentation'][0], dim=0).numpy(),
        cmap="viridis"
    )
    axs[i, 1].set_title("Segmentation")
plt.tight_layout()
plt.savefig("Section 2_7.png")

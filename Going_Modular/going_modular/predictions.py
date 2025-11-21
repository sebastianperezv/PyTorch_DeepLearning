from typing import List, Tuple

from PIL import Image

from torchvision import transforms

# 1. Take in a trained model
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device = device):
  # 2. Open the image with PIL
  img = Image.open(image_path)

  # 3. Create a transform if ione does not exist
  if transform is not None:
    image_transform = transform
  else:
    image_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

  ### Predict on image ###
  # 4. Make sure the model is on the target device
  model.to(device)

  # 5. Turn on inference mode and eval mode
  model.eval() # will turn off things like droput
  with torch.inference_mode():
    # 6. Transform the image and add an extra batch dimension
    # our model expect a batch dimension
    transformed_image = image_transform(img).unsqueeze(dim=0) # shape of [batch_size, color_channels, height, width]

    # 7. Make a prediction on the transformed image by passing it ti the model (also ensure it's on the target device)
    target_image_pred = model(transformed_image.to(device)) # this is an output logits

  # 8. Convert the model's output logits to pred probs
  target_image_pred_probs = torch.softmax(target_image_pred, dim = 1)
  print(target_image_pred_probs)

  # 9. Conver the model's pred probs to pred label
  target_image_pred_label = torch.argmax(target_image_pred_probs, dim = 1)

  # 10. Plot iamge with predicted label probability
  plt.figure()
  plt.imshow(img)
  plt.title(f'Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}')
  plt.axis(False)

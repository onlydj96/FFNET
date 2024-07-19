
import torchvision.transforms as transforms
import numpy as np

def image_preprocess(image, fixed_size=(512, 512)):
    transform = transforms.Compose([
        transforms.Resize(fixed_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).numpy()
    return image

def pred_heatmap(session, image, fixed_size=(1024, 1024)): #, threshold=0.5):
    image = image_preprocess(image, fixed_size)
    inputs = {session.get_inputs()[0].name: image}
    pred_map = session.run(None, inputs)[0]
    
    pred_map = pred_map.squeeze(0).squeeze(0)
    
    # Threshold 적용
    # pred_map = np.where(pred_map > threshold, pred_map, 0)
    
    return pred_map
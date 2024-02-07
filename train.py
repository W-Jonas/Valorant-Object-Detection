from ultralytics import YOLO
import torch
import torchvision
from multiprocessing import freeze_support

# Check if CUDA GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Running on GPU')
else:
    device = torch.device('cpu')
    print('Running on CPU')

if __name__ == '__main__':
    freeze_support()

    print(torchvision.__version__)

    input('Press ENTER to start training...')

    # Load a model
    model = YOLO('yolov8n.yaml').to(device)

    # Train the model
    results = model.train(data='dataset\data.yaml', epochs=100, device=0, batch=-1)
    model.export(format="TensorRT", dynamic=True)

from torchvision.transforms import v2
import cv2

train_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        v2.GaussianNoise(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        v2.RandomPerspective(),
        v2.ElasticTransform(),
        v2.GaussianBlur(),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        v2.Grayscale(),
        v2.Lambda(lambda x: cv2.Canny(x, 100, 200)),
    ]
)

test_transform = v2.Compose(
    [
        v2.toImage(),
        v2.GaussianBlur(),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        v2.Grayscale(),
        v2.Lambda(lambda x: cv2.Canny(x, 100, 200)),
    ]
)

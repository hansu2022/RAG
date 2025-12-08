import torch
from torchvision import transforms
from torchvision.transforms import v2
import logging
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class ImageBindInterface():
    def __init__(self, device=device):
        self.device = device
        logging.info("读取模型权重中...")
        self.model = imagebind_model.imagebind_huge(pretrained=True)        
        self.model.eval()
        self.model.to(device)
        logging.info("模型加载完毕")
        self.img_trans = v2.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def process_text(self, text):
        if isinstance(text, str):
            text = [text]
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(text, device)
        }
        return inputs
    
    def process_images(self, image_mats):
        # Input image: (...,H, W, C)
        if not isinstance(image_mats, list):
            image_mats = [image_mats]
        images = []
        for img in image_mats:
            images.append(self.img_trans(img).to(device))
        images = torch.stack(images, dim=0)
        inputs = {
            ModalityType.VISION: images
        }
        return inputs
    
    def embed_input(self, modality_dict):
        with torch.no_grad():
            embeddings = self.model(modality_dict)
        return embeddings
    
    def unwrap_output(self, output_dict):
        return [value for key, value in output_dict.items()]


if __name__ == "__main__":
    from moviepy import VideoFileClip
    res = data.load_and_transform_video_data(video_paths=["/data/rag/video/data/1_2_1.mp4"], device=device)
    print(res.shape)
    """test_text = ["Hi", "How are you?"]
    video = VideoFileClip("/data/rag/video/data/1_2_1.mp4")
    test_frame = [video.get_frame(0.0), video.get_frame(5.0)]
    
    model = ImageBindInterface()
    out = model.embed_input({**model.process_text(test_text), **model.process_images(test_frame)})
    print(out)
    out = model.unwrap_output(out)
    print(out[0])
    print(out[1])"""
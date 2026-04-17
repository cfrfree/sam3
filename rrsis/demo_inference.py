import argparse
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T
from PIL import Image
from scipy.ndimage.morphology import binary_dilation
from bert.tokenization_bert import BertTokenizer
from bert.modeling_bert import BertModel
from lib import segmentation

# ================= 配置区域 =================
DEFAULTS = {
    "image_path": "/home/share/chenfree/refship/images/P0077_600_1400_3600_4400.jpg",
    "sentence": "the 2 offshore ships at the bottom",
    "weights": "/home/chenfree2002/Python/TGRS/rrsis/logs/model_best_1234.pth",
    "bert_path": "/home/chenfree2002/Python/TGRS/refseg/checkpoints/bert-base-uncased",
    "device": "cuda:0",
    "output_path": "demo_result.jpg",
}


class ModelArgs:
    """模拟 config 文件中的参数类"""

    swin_type = "base"  # 请确保这里和训练时的 swin_type 一致 (base/tiny/small/large)
    window12 = True  # 训练时如果用了 window12，这里保持 True
    mha = ""
    fusion_drop = 0.0


def get_args():
    parser = argparse.ArgumentParser(description="Inference Script")
    parser.add_argument(
        "--image_path", default=DEFAULTS["image_path"], help="Path to input image"
    )
    parser.add_argument("--sentence", default=DEFAULTS["sentence"], help="Text prompt")
    parser.add_argument(
        "--weights", default=DEFAULTS["weights"], help="Path to model weights (.pth)"
    )
    parser.add_argument(
        "--bert_path", default=DEFAULTS["bert_path"], help="Path to BERT checkpoint"
    )
    parser.add_argument(
        "--device", default=DEFAULTS["device"], help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--output_path", default=DEFAULTS["output_path"], help="Path to save result"
    )
    return parser.parse_args()


def overlay_davis(image, mask, colors=[[0, 0, 0], [255, 0, 0]], cscale=1, alpha=0.4):
    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale
    im_overlay = image.copy()
    object_ids = np.unique(mask)
    for object_id in object_ids[1:]:
        foreground = image * alpha + np.ones(image.shape) * (1 - alpha) * np.array(
            colors[object_id]
        )
        binary_mask = mask == object_id
        im_overlay[binary_mask] = foreground[binary_mask]
        contours = binary_dilation(binary_mask) ^ binary_mask
        im_overlay[contours, :] = 0
    return im_overlay.astype(image.dtype)


def main():
    args = get_args()
    device = torch.device(args.device)

    print(f"Loading image from: {args.image_path}")
    print(f"Prompt: {args.sentence}")

    # 1. Data Pre-processing (Image)
    raw_img = Image.open(args.image_path).convert("RGB")
    original_w, original_h = raw_img.size
    img_ndarray = np.array(raw_img)

    # 【修改点 1】: 建议强制 Resize 到 (480, 480) 而不是单边 480
    # 你的 train.py 中使用的是 T.Resize(args.img_size, args.img_size)，即强制方形。
    # 如果只写 T.Resize(480)，长宽比会保持，导致输入变成 (480, 640) 等，Swin Transformer 可能因为 window size 对不齐而报错或性能下降。
    image_transforms = T.Compose(
        [
            T.Resize((480, 480)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img_tensor = image_transforms(raw_img).unsqueeze(0).to(device)

    # 2. Data Pre-processing (Text)
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    sentence_tokenized = tokenizer.encode(text=args.sentence, add_special_tokens=True)
    max_len = 20
    sentence_tokenized = sentence_tokenized[:max_len]
    padded_sent_toks = [0] * max_len
    padded_sent_toks[: len(sentence_tokenized)] = sentence_tokenized
    attention_mask = [0] * max_len
    attention_mask[: len(sentence_tokenized)] = [1] * len(sentence_tokenized)

    padded_sent_toks = torch.tensor(padded_sent_toks).unsqueeze(0).to(device)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)

    # 3. Model Initialization
    print("Initializing model...")
    model = segmentation.__dict__["lavt"](pretrained="", args=ModelArgs)
    model.to(device)

    bert_model = BertModel.from_pretrained(args.bert_path)
    bert_model.pooler = None
    bert_model.to(device)

    # 4. Load Weights (【关键修改点 2】)
    try:
        print(f"Loading weights from: {args.weights}")
        # 加载整个 checkpoint (包含 epoch, optimizer, model, bert_model 等)
        checkpoint = torch.load(args.weights, map_location=device)

        # 1. 加载 LAVT 模型权重
        if "model" in checkpoint:
            print("Loading LAVT weights...")
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            print("Warning: 'model' key not found in checkpoint.")

        # 2. 加载 BERT 模型权重 (这步非常重要！)
        if "bert_model" in checkpoint:
            print("Loading BERT weights...")
            bert_model.load_state_dict(checkpoint["bert_model"], strict=False)
        else:
            print(
                "Warning: 'bert_model' key not found in checkpoint! Using pre-trained BERT (results might be poor)."
            )

        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.eval()
    bert_model.eval()

    # 5. Inference
    with torch.no_grad():
        last_hidden_states = bert_model(
            padded_sent_toks, attention_mask=attention_mask
        )[0]
        embedding = last_hidden_states.permute(0, 2, 1)  # (B, Hidden, Seq_len)

        output = model(img_tensor, embedding, l_mask=attention_mask.unsqueeze(-1))

        output = output.argmax(1, keepdim=True)
        # 恢复到原始尺寸
        output = F.interpolate(output.float(), (original_h, original_w), mode="nearest")
        output = output.squeeze().cpu().numpy()

    # 6. Visualization
    output = output.astype(np.uint8)
    visualization = overlay_davis(img_ndarray, output)

    vis_img = Image.fromarray(visualization)
    vis_img.save(args.output_path)
    print(f"Result saved to {args.output_path}")


if __name__ == "__main__":
    main()

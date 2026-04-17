import os
import torch
import torch.nn.functional as F
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import binary_dilation

# 导入你的模块
from refer.refer import REFER
from bert.modeling_bert import BertModel
from bert.tokenization_bert import BertTokenizer
from lib import segmentation
import transforms as T


# ================= 1. 配置区域 =================
class Config:
    # 路径配置
    dataset_root = "/home/chenfree2002/share/chenfree/refship"
    instances_json = "/home/share/chenfree/refship/instances.json"
    weights_path = "/home/chenfree2002/Python/TGRS/rrsis/logs/model_best_1234.pth"
    bert_path = "/home/chenfree2002/Python/TGRS/refseg/checkpoints/bert-base-uncased"
    output_dir = "/home/chenfree2002/share/chenfree/refship/vis_new/LGCE"
    
    device = "cuda:0"
    img_size = 480
    id_offset = 6765 # 定义偏移量: json_id = seg_id + 6765

    # 模型相关参数（需与训练时一致）
    swin_type = "base"
    window12 = True
    mha = ""
    fusion_drop = 0.0


# ================= 2. 工具函数 =================
def overlay_davis(image, mask, colors=[[0, 0, 0], [255, 0, 0]], alpha=0.4):
    colors = np.reshape(colors, (-1, 3))
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
    return im_overlay.astype(np.uint8)


def main():
    cfg = Config()
    device = torch.device(cfg.device)

    # 1. 读取 instances.json 并建立 id -> file_name 映射
    print(f"Loading mapping from {cfg.instances_json}...")
    with open(cfg.instances_json, 'r') as f:
        instances_data = json.load(f)
    # 建立查找表：{id: file_name}
    id_to_filename = {img['id']: img['file_name'] for img in instances_data['images']}

    # 2. 初始化 REFER 接口 (获取 train 和 val)
    refer = REFER(cfg.dataset_root, dataset="refship", splitBy="unc")
    ref_ids = refer.getRefIds(split="train") + refer.getRefIds(split="val")
    print(f"Total references to process: {len(ref_ids)}")

    # 3. 准备输出目录
    mask_dir = os.path.join(cfg.output_dir, "masks")
    vis_dir = os.path.join(cfg.output_dir, "visualizations")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # 4. 初始化模型与分词器
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_path)
    model = segmentation.__dict__["lavt"](pretrained="", args=cfg)
    model.to(device)
    bert_model = BertModel.from_pretrained(cfg.bert_path)
    bert_model.pooler = None
    bert_model.to(device)

    # 5. 加载权重
    print(f"Loading weights from {cfg.weights_path}...")
    checkpoint = torch.load(cfg.weights_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    if "bert_model" in checkpoint:
        bert_model.load_state_dict(checkpoint["bert_model"], strict=False)

    model.eval()
    bert_model.eval()

    # 6. 图像预处理定义
    image_transforms = T.Compose(
        [
            T.Resize(cfg.img_size, cfg.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 7. 开始遍历推理
    for ref_id in tqdm(ref_ids, desc="LGCE Inference"):
        ref = refer.loadRefs(ref_id)[0]
        image_info = refer.loadImgs(ref["image_id"])[0]

        # ================= 关键修改：计算并查找原始文件名 =================
        seg_id = ref['ann_id'] # 获取实例 ID
        target_id = int(seg_id) + cfg.id_offset
        if target_id in id_to_filename:
            # 使用 json 中对应的原始文件名，例如 "000689.jpg" -> "000689"
            orig_file_base = os.path.splitext(id_to_filename[target_id])[0]
        else:
            # 兜底：使用默认文件名
            orig_file_base = os.path.splitext(image_info['file_name'])[0]
        # ==============================================================

        # 加载并处理图片
        img_path = os.path.join(refer.IMAGE_DIR, image_info['file_name'])
        if not os.path.exists(img_path): continue
            
        raw_img = Image.open(img_path).convert("RGB")
        original_w, original_h = raw_img.size
        
        # 创建空白占位图解决 transforms.py 的参数要求
        placeholder_target = Image.new('L', (original_w, original_h))
        
        img_tensor, _ = image_transforms(raw_img, placeholder_target)
        img_tensor = img_tensor.unsqueeze(0).to(device)
        img_ndarray = np.array(raw_img)

        # 遍历该目标的所有描述语句
        for sent_info in ref["sentences"]:
            sentence = sent_info["sent"]
            sent_id = sent_info["sent_id"]

            # 文本预处理
            tokens = tokenizer.encode(
                text=sentence, add_special_tokens=True, max_length=20, truncation=True
            )
            padded_tokens = torch.zeros(1, 20).long().to(device)
            padded_tokens[0, : len(tokens)] = torch.tensor(tokens)
            attn_mask = torch.zeros(1, 20).long().to(device)
            attn_mask[0, : len(tokens)] = 1

            # 推理
            with torch.no_grad():
                last_hidden_states = bert_model(
                    padded_tokens, attention_mask=attn_mask
                )[0]
                embedding = last_hidden_states.permute(0, 2, 1)
                output = model(img_tensor, embedding, l_mask=attn_mask.unsqueeze(-1))

                # 后处理
                pred_mask = output.argmax(1, keepdim=True)
                pred_mask = F.interpolate(
                    pred_mask.float(), (original_h, original_w), mode="nearest"
                )
                pred_mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8)

            # 最终命名规则: 原始图片名 + 描述ID
            safe_sent_id = str(sent_id).replace("/", "_")
            base_filename = orig_file_base

            # 保存 Mask
            Image.fromarray(pred_mask * 255).save(
                os.path.join(mask_dir, f"{base_filename}.png")
            )

            # 保存叠加效果图
            vis_img = overlay_davis(img_ndarray, pred_mask)
            Image.fromarray(vis_img).save(
                os.path.join(vis_dir, f"{base_filename}.jpg")
            )

    print(f"All done! Results saved to {cfg.output_dir}")


if __name__ == "__main__":
    main()
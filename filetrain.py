import os
import torch
from PIL import Image
from google.colab import drive
from datasets import load_dataset
from transformers import TextStreamer
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# ==========================================
# 1. KẾT NỐI DRIVE & GIẢI NÉN
# ==========================================
drive.mount('/content/drive')
ZIP_PATH = "/content/drive/MyDrive/VLM_Project/train.zip"
EXTRACT_DIR = "/content/train_data"

print("Đang giải nén dữ liệu...")
!unzip -q -o {ZIP_PATH} -d {EXTRACT_DIR}

JSONL_FILE = "/content/train_data/train/metadata.jsonl"
IMAGES_DIR = "/content/train_data/train"

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH LORA (Qwen2-VL-7B + Ép GPU)
# ==========================================
print("Đang tải mô hình...")
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2-VL-7B-Instruct",
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
    device_map = {"": 0} # Trói chặt vào GPU T4
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = 32,
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

# ==========================================
# 3. LOAD DATASET (Dùng .map() chống tràn RAM)
# ==========================================
print(f"Đọc dữ liệu từ: {JSONL_FILE}")
raw_dataset = load_dataset("json", data_files=JSONL_FILE, split="train")

instruction = """You are an expert AI in image analysis. Detect all persons in the image, extract their bounding boxes [ymin, xmin, ymax, xmax], and analyze their interactions (e.g., talking, face-to-face). Output strictly in valid JSON format. IMPORTANT: The values for 'state', 'talking', and 'interaction' MUST be in Vietnamese (e.g., "không xác định", "có", "đang trò chuyện")."""

def format_dataset(sample):
    image_path = os.path.join(IMAGES_DIR, sample["file_name"])
    image = Image.open(image_path).convert("RGB")
    
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "image", "image" : image},
            {"type" : "text",  "text"  : instruction}
          ] 
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["ground_truth"]} 
          ]
        },
    ]
    return { "messages" : conversation }

print("Đang xử lý map dataset (Lazy-load chống treo máy)...")
converted_dataset = raw_dataset.map(format_dataset, num_proc=1, remove_columns=raw_dataset.column_names)
print(f"✅ Hoàn tất load {len(converted_dataset)} mẫu dữ liệu!")

# ==========================================
# 4. BẮT ĐẦU TRAIN
# ==========================================
FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 1,     
        gradient_accumulation_steps = 8,     
        num_train_epochs = 3,                
        learning_rate = 2e-4,
        lr_scheduler_type = "cosine",
        warmup_ratio = 0.05,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 1,
        max_seq_length = 2048,
    ),
)

print("🚀 Bắt đầu quá trình Finetune...")
trainer_stats = trainer.train()
print("✅ Finetune hoàn tất!")

# ==========================================
# 5. LƯU MÔ HÌNH VỀ DRIVE 
# ==========================================
SAVE_DIR = "/content/drive/MyDrive/VLM_Project/Saved_Model"
print(f"💾 Đang lưu model vào Drive: {SAVE_DIR}...")
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print("🎉 ĐÃ XONG TẤT CẢ!")
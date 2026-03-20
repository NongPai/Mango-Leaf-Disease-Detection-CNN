import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# CLIP
from transformers import CLIPProcessor, CLIPModel
import torch

# โหลดโมเดลโรคใบมะม่วง (EfficientNet)
model = tf.keras.models.load_model("Enter Your Model Name")

# โหลดโมเดล CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# รายชื่อคลาสโรค (อังกฤษ)
class_names = [
    'Anthracnose', 'BacterialCanker', 'CuttingWeevil', 'DieBack',
    'GallMidge', 'Healthy', 'NotMangoLeaf', 'PowderyMildew', 'SootyMould'
]

# พจนานุกรมแปลชื่อโรค (อังกฤษ -> ไทย)
class_name_thai = {
    'Anthracnose': 'โรคแอนแทรคโนส',
    'BacterialCanker': 'โรคแบคทีเรียแคงเกอร์',
    'CuttingWeevil': 'แมลงปีกแข็งทำลายใบ',
    'DieBack': 'โรคใบไหม้แห้ง',
    'GallMidge': 'เพลี้ยไฟมะม่วง',
    'Healthy': 'ใบมะม่วงปกติ',
    'NotMangoLeaf': 'ไม่ใช่ใบมะม่วง',
    'PowderyMildew': 'โรคราแป้ง',
    'SootyMould': 'โรคราดำ'
}

# พจนานุกรมแนะนำวิธีจัดการโรค
disease_advice = {
    'Anthracnose': "ควรตัดแต่งใบที่เป็นโรคออก และพ่นสารป้องกันเชื้อรา เช่น แมนโคเซบ",
    'BacterialCanker': "ควรตัดแต่งกิ่งที่ติดเชื้อ และใช้สารกำจัดแบคทีเรีย เช่น คอปเปอร์ออกซีคลอไรด์",
    'CuttingWeevil': "ใช้สารฆ่าแมลง และกำจัดเศษซากพืชรอบโคนต้น",
    'DieBack': "ตัดแต่งกิ่งที่แห้งออก และพ่นสารป้องกันเชื้อรา",
    'GallMidge': "ใช้สารกำจัดแมลงปีกแข็ง และดูแลความสะอาดพื้นที่ปลูก",
    'Healthy': "ไม่พบโรค ดูแลรักษาต้นมะม่วงตามปกติ เช่น ให้น้ำและปุ๋ยอย่างเหมาะสม",
    'NotMangoLeaf': "ไม่ใช่ใบมะม่วง ไม่สามารถให้คำแนะนำได้",
    'PowderyMildew': "พ่นสารกำจัดเชื้อรา เช่น ซัลเฟอร์ หรือไตรฟลอกซี่สโตรบิน",
    'SootyMould': "ล้างใบด้วยน้ำสะอาด และกำจัดเพลี้ยหรือแมลงที่เป็นพาหะ"
}

# ฟังก์ชันตรวจสอบว่าใช่ใบมะม่วงหรือไม่ ด้วย CLIP
def is_mango_leaf(img):
    inputs = clip_processor(
        text=["a photo of a mango leaf", "a photo of something else"],
        images=img,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return probs[0][0].item() >= 0.70  # Threshold 70%

# ฟังก์ชันทำนายโรคใบมะม่วง
def predict_disease(img):
    try:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((224, 224))

        # ตรวจสอบว่าเป็นใบมะม่วงไหม
        if not is_mango_leaf(img):
            return "ไม่ใช่ใบมะม่วง", "N/A", "ไม่สามารถให้คำแนะนำได้"

        # Predict with EfficientNet
        img_array = image.img_to_array(img)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        top_idx = np.argmax(preds)
        confidence = float(preds[0][top_idx])

        # มั่นใจน้อยกว่า 70% ให้บอกว่าไม่ใช่ใบมะม่วง
        if confidence < 0.7:
            return "ไม่ใช่ใบมะม่วง", f"{confidence * 100:.2f}%", "ไม่สามารถให้คำแนะนำได้"

        predicted_class = class_names[top_idx]
        predicted_class_thai = class_name_thai.get(predicted_class, predicted_class)
        advice = disease_advice.get(predicted_class, "ไม่มีคำแนะนำ")

        return predicted_class_thai, f"{confidence * 100:.2f}%", advice

    except Exception as e:
        print("🔥 ERROR:", e)
        return "ข้อผิดพลาด", "ข้อผิดพลาด", "ข้อผิดพลาด"

# Gradio UI
interface = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(type="pil", label="📷 อัปโหลดภาพใบมะม่วง"),
    outputs=[
        gr.Textbox(label="🩺 ผลการทำนาย (ชื่อโรค)"),
        gr.Textbox(label="📈 ความมั่นใจ (%)"),
        gr.Textbox(label="💡 คำแนะนำในการจัดการโรค")
    ],
    title="🍃 ระบบวิเคราะห์โรคใบมะม่วงด้วย AI",
    description="อัปโหลดรูปภาพใบมะม่วง ระบบจะตรวจสอบว่าใช่ใบมะม่วงหรือไม่ และทำนายว่าใบมีโรคหรือไม่ พร้อมแนะนำวิธีดูแลรักษา",
    theme="soft",
    examples=[
        ["example_images/Example_1.jpg"],
        ["example_images/Example_2.jpg"],
        ["example_images/Example_3.jpg"]
    ]
)

# Run App
if __name__ == "__main__":
    interface.launch()
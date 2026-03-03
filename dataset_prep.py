import os
import torch
from PIL import Image, ImageOps
from transformers import BlipProcessor, BlipForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

proccesor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model=BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

image_folder = r"C:\Users\Berk\OneDrive\Desktop\Drone_Dataset"
base_filename = "drone"
counter = 0

files = sorted(os.listdir(image_folder))

for filename in files:
    if filename.lower().endswith(('.png','.jpg','.jpeg')):
        old_image_path = os.path.join(image_folder,filename)
        
        new_file_name = f"{base_filename}_{counter:03d}.jpg"
        new_image_path = os.path.join(image_folder,new_file_name)
        
        raw_image= Image.open(old_image_path).convert("RGB")
        
        resized_image= ImageOps.fit(raw_image, (512,512),method = Image.LANCZOS,centering = (0.5,0.5))
        
        inputs = proccesor(resized_image,return_tensors = "pt").to(device)
        
        with torch.no_grad():
            outs = model.generate(**inputs,min_new_tokens = 20)
            
        caption = proccesor.decode(outs[0],skip_special_tokens = True)
        final_caption =f"berkdrone, {caption}, a white DJI Phantom 3 drone, product photography, highly detailed, 8k resolution"
        
        resized_image.save(new_image_path,quailt=95)
        
        txt_filename = f"{base_filename}_{counter:03d}.txt"
        txt_path = os.path.join(image_folder,txt_filename)
        
        with open(txt_path,"w",encoding = "utf-8") as f:
            f.write(final_caption)
        
        counter += 1
            
        
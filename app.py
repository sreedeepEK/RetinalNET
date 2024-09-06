import gradio as gr
import os 
import torch 

from src.model import resnet_model
from timeit import default_timer as timer
from typing import Tuple,Dict

class_names = ["CNV","DME","DRUSEN","NORMAL"]

resnet, resnet_transforms = resnet_model(num_classes=4)


state_dict = torch.load(f="models/model.pth", map_location=torch.device("cpu"))
resnet.load_state_dict(state_dict, strict=False)

def predict(img) -> Tuple[Dict,float]:
    start_time = timer()
    
    img = resnet_transforms(img).unsqueeze(0)
    resnet.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(resnet(img),dim=1)
        
    pred_label_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    pred_time = round(timer() - start_time,5)
    return pred_label_and_probs, pred_time

example_paths = [["examples/" + example] for example in os.listdir("examples")]

title = "Retinal disease detection using Optical Tomography Images 👁️"
description = " This application uses Optical Coherence Tomography (OCT) images to assist in the identification of retinal conditions such as CNV, DME, DRUSEN, and NORMAL. The tool provides predictions based on the uploaded image and displays the processing time for the analysis. Please note that this tool is intended for educational and research purposes only. It is not a substitute for professional medical advice or diagnosis. For any medical concerns, please consult a healthcare professional."

gradio_interface = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=4,label="Predictions"),
                                gr.Number(label="prediction time: ")],
                                title=title,
                                examples=example_paths,
                                description=description)
                            
gradio_interface.launch()
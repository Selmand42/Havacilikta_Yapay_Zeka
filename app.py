import gradio as gr
from ultralytics import YOLO
import torch

desc = """
**TEKNOFEST Artificial Intelligence in Aviation Competition**

This test is built on the **YOLOv8 architecture** and further improved through **fine-tuning**. Developed within the scope of the TEKNOFEST Artificial Intelligence in Aviation Competition, it is designed to perform object detection and classification on aerial imagery with high accuracy.

## Key Features
- YOLOv8-based modern object detection  
- Fine-tuned on competition-specific datasets  
- Optimized for aviation scenarios  
"""

model = YOLO("model.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def predict(image):
    results = model(image, device=device)
    res_plotted = results[0].plot()
    return res_plotted

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="QuadAI YOLOv8",
    description=desc,
    examples=[
        ["Frames/frame1.jpg"],
        ["Frames/frame2.jpg"],
        ["Frames/frame3.jpg"],
        ["Frames/frame4.jpg"],
        ["Frames/frame5.jpg"]
    ]
)

demo.launch()

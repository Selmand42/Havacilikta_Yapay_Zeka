import gradio as gr
from ultralytics import YOLO
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model = YOLO("modelv12.pt")

def predict(image):
    results = model(image, device=device)  # inference cihazını belirt
    res_plotted = results[0].plot()
    return res_plotted

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="YOLO Detection",
    description="Upload an image and get YOLO detection results",
    examples=[
        ["Frames/frames1.jpg"],
        ["Frames/frames2.jpg"],
        ["Frames/frames3.jpg"],
        ["Frames/frames4.jpg"],
        ["Frames/frames5.jpg"]
    ]
)

demo.launch()

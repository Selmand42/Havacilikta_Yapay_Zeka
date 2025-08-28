import gradio as gr
from ultralytics import YOLO
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model = YOLO("model.pt")

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
        ["Frames/frame1.jpg"],
        ["Frames/frame2.jpg"],
        ["Frames/frame3.jpg"],
        ["Frames/frame4.jpg"],
        ["Frames/frame5.jpg"]
    ]
)

demo.launch()

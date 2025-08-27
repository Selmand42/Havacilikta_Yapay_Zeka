import gradio as gr
from ultralytics import YOLO

# Modeli yükle
model = YOLO("model.pt")

def predict(image):
    results = model(image)
    res_plotted = results[0].plot()  # Kutular çizilmiş numpy array
    return res_plotted  # cv2.cvtColor ile dönüştürme yok

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="YOLO Detection",
    description="Upload an image and get YOLO detection results"
)

demo.launch()

from flask import Flask, request, render_template
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

image_size = 128
class_names = ['Dresses', 'Heels', 'Jeans', 'Sandals', 'Shorts', 'Tshirts']

# 🧠 Define CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * (image_size // 8) * (image_size // 8), 512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(class_names))
        )
    def forward(self, x):
        return self.model(x)

# 🔁 Load model ONCE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(len(class_names)).to(device)
model.load_state_dict(torch.load('product_classifier.pth', map_location=device))
model.eval()
print("✅ Model loaded successfully.")


# 🧽 Preprocessing
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

@app.route('/', methods=['GET', 'POST'])
def index():
    image_path = None
    prediction = None

    if request.method == 'POST':
        if 'image' in request.files:
            file = request.files['image']
            if file:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                print("✅ Image uploaded:", filepath)
                return render_template('index.html', image_path=filepath)

        elif 'classify' in request.form:
            image_path = request.form.get('image_path')
            print("👉 Classifying:", image_path)

            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = model(img_tensor)
                        _, predicted = torch.max(outputs, 1)
                        prediction = class_names[predicted.item()]
                        print("✅ Prediction:", prediction)

                    return render_template('index.html', image_path=image_path, prediction=prediction)

                except Exception as e:
                    print("❌ Classification error:", str(e))
                    return "Error during prediction", 500
            else:
                print("❌ Image not found:", image_path)
                return "Image not found", 404

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)  # Required for Render

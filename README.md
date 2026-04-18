# XAI Course Project — Explainable AI with SHAP, LIME & Grad-CAM

A full-stack web application that classifies images, text, tabular data, and audio using deep neural networks
and explains **why** they made that prediction using multiple XAI techniques.

---

## What It Does

Upload an image, enter text, input tabular data, or upload audio → the system:
1. **Predicts** the class using appropriate deep learning models
2. **Explains with SHAP** — shows feature importance using game theory
3. **Explains with LIME** — highlights influential parts using perturbations
4. **Explains with Grad-CAM** — shows CNN focus areas (for images)

---

## Tech Stack

| Layer     | Technology                          |
|-----------|-------------------------------------|
| Models    | CNN (CIFAR-10), LSTM (IMDB), DNN (Iris), 1D-CNN (Audio) |
| Backend   | FastAPI (Python)                    |
| Frontend  | React                               |
| XAI       | SHAP, LIME, Grad-CAM                |

---

## Project Structure

```
backend/
  app.py          ← FastAPI routes (/predict/*)
  models.py       ← Model loading helpers
  explainers.py   ← SHAP, LIME, Grad-CAM classes
  data.py         ← Preprocessing and data helpers
  requirements.txt

frontend/
  src/
    App.jsx                        ← Main app, API calls
    components/
      ImageModel.jsx               ← Image upload UI
      TextModel.jsx                ← Text input UI
      TabularModel.jsx             ← Form for Iris features
      AudioModel.jsx               ← Audio upload UI
      ResultPanel.jsx              ← Prediction results
      ExplanationView.jsx          ← XAI explanations display
```

---

## Note on Predictions

While the models are trained on standard datasets and provide accurate predictions in many cases,
no AI model is perfect. The explanations help users understand the model's reasoning,
identify potential biases, and build trust in the predictions.

---

## How to Run

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

API will be available at `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs`

> **Optional:** Fine-tune the model on CIFAR-10 first (improves accuracy):
> ```bash
> python train.py
> ```
> This saves `saved_model.pt`. Without it, the app uses pretrained ImageNet weights.

### 2. Frontend

```bash
cd frontend
npm install
npm start
```

App opens at `http://localhost:3000`

---

## XAI Methods Explained

### LIME (Local Interpretable Model-agnostic Explanations)
- Divides the image into segments (superpixels)
- Creates ~500 perturbed versions by randomly masking segments
- Runs each through the model and records predictions
- Fits a simple linear model to find which segments matter most
- **Model-agnostic** — works with any classifier

### Grad-CAM (Gradient-weighted Class Activation Mapping)
- Runs a forward pass and records feature maps at ResNet18's `layer4`
- Runs a backward pass for the predicted class to get gradients
- Averages gradients spatially → importance weight per feature map channel
- Weighted sum of feature maps → coarse spatial heatmap
- Applies ReLU (keep only positive contributions) and normalizes
- **CNN-specific** — requires access to gradients

---

## API Endpoints

| Method | Endpoint          | Description                        |
|--------|-------------------|------------------------------------|
| GET    | `/he on http://0.0.0.0:8000
```

✅ Backend is running! Visit http://localhost:8000/docs to see API documentation.

### Step 3: Frontend Setup

```bash
# Open NEW terminal, navigate to frontend
cd frontend

# Install dependencies
npm install
```

### Step 4: Start Frontend

```bash
# In frontend folder
npm start
```

You should see:
```
Compiled successfully!
Local: http://localhost:3000
```

✅ Frontend is running! Open http://localhost:3000 in your browser.

---

## 🚀 How to Use

1. **Open Application**
   - Frontend: http://localhost:3000

2. **Upload Image**
   - Click on upload area
   - Select a CIFAR-10 image (or any small image)

3. **Get Prediction**
   - Click "Predict & Explain" button
   - Wait for processing (3-5 seconds)

4. **View Results**
   - See predicted class and confidence
   - See LIME explanation (important regions)
   - See Grad-CAM explanation (activated neurons)

---

## 📚 Training Your Own Model

To train a custom model:

```bash
cd backend
python train.py
```

This will:
- Load CIFAR-10 dataset (downloads automatically)
- Train ResNet18 for 10 epochs
- Save best model as `saved_model.pt`
- Display accuracy and loss for each epoch

Expected training time: 5-10 minutes (CPU), 1-2 minutes (GPU)

---

## 🔌 API Endpoints

### Health Check
```
GET /health
```

### Make Prediction
```
POST /predict
Content-Type: multipart/form-data

Body: image file
```

Response:
```json
{
  "prediction": "cat",
  "confidence": 0.95,
  "all_predictions": [
    {"class": "cat", "probability": 0.95},
    {"class": "dog", "probability": 0.04}
  ]
}
```

### Get LIME Explanation
```
POST /explain-lime
Content-Type: multipart/form-data

Body: image file
```

### Get Grad-CAM Explanation
```
POST /explain-gradcam
Content-Type: multipart/form-data

Body: image file
```

### Get Classes
```
GET /classes
```

### Get Info
```
GET /info
```

---

## 🔧 Troubleshooting

### Backend won't start
```bash
# Check Python version (need 3.8+)
python --version

# Reinstall requirements
pip install -r requirements.txt --upgrade

# Try different port
# Edit app.py, change port 8000 to 8001
```

### CORS errors
- Make sure backend is running on http://localhost:8000
- Make sure frontend is on http://localhost:3000
- Check .env file has correct API URL

### Port already in use
```bash
# Find process using port 8000
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill process
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows
```

### Model download fails
```bash
# Models download automatically on first run
# If it fails, try manually:
python -c "import torchvision; torchvision.models.resnet18(pretrained=True)"
```

---

## 📖 Understanding the Code

### Backend Flow

```
Request (image)
    ↓
app.py - receives request
    ↓
data.py - preprocess image
    ↓
models.py - predict
    ↓
explainers.py - generate LIME/Grad-CAM
    ↓
Response (prediction + explanations)
```

### Frontend Flow

```
User Interface
    ↓
App.jsx - main state management
    ↓
PredictForm.jsx - upload image
    ↓
Axios - call backend API
    ↓
PredictionResult.jsx - show prediction
    ↓
ExplanationView.jsx - show explanations
```

---

## 📝 What is LIME?

**LIME (Local Interpretable Model-agnostic Explanations)**
- Explains why model made a specific prediction
- Works by perturbing the image and observing changes
- Shows which regions are important for the prediction
- Model-agnostic (works with any model)

---

## 📝 What is Grad-CAM?

**Grad-CAM (Gradient-weighted Class Activation Mapping)**
- Shows which parts of the image activated the neurons
- Uses gradients to weight feature maps
- Creates a heatmap of important regions
- CNN-specific but efficient

---

## 🎓 Learning Outcomes

After completing this project, you'll understand:

✅ How to build deep learning models with PyTorch
✅ How to create REST APIs with FastAPI
✅ How to build web UIs with React
✅ What LIME and Grad-CAM explanations are
✅ How to make models interpretable
✅ Full-stack development (frontend + backend)

---

## 📊 Project Requirements (Course)

This project covers:

- [x] Build a machine learning model (ResNet18 CNN)
- [x] Train on a dataset (CIFAR-10)
- [x] Make predictions on new data
- [x] Implement 2 explanation methods (LIME + Grad-CAM)
- [x] Create a user interface
- [x] Document the work

**Total Implementation Time**: 2-4 weeks
**Total Code**: ~1,600 lines
**Difficulty**: Beginner to Intermediate

---

## 🐛 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError | Install missing package: `pip install <package>` |
| Address already in use | Change port in code or kill existing process |
| CORS error | Check backend URL in frontend .env file |
| Slow prediction | Use GPU (CUDA) or reduce image size |
| Model not found | Train model: `python train.py` |

---

## 📚 File Descriptions

| File | Lines | Purpose |
|------|-------|---------|
| app.py | 200+ | FastAPI server with endpoints |
| models.py | 30 | CNN model definition |
| data.py | 90 | Data loading and preprocessing |
| explainers.py | 150+ | LIME and Grad-CAM implementations |
| train.py | 100+ | Model training script |
| App.jsx | 120+ | Main React component |
| PredictForm.jsx | 60+ | File upload component |
| PredictionResult.jsx | 70+ | Results display component |
| ExplanationView.jsx | 100+ | Explanation display component |
| Navbar.jsx | 30+ | Navigation component |

**Total: ~1,600 lines of code**

---

## 🔗 Resources

### Documentation
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [React Docs](https://react.dev/)
- [PyTorch Docs](https://pytorch.org/docs/)
- [LIME GitHub](https://github.com/marcotcr/lime)

### CIFAR-10 Dataset
- [Download](https://www.cs.toronto.edu/~kriz/cifar.html)
- 60,000 images, 10 classes
- Auto-downloads when training

---

## 💡 Tips

1. **Start with the backend first** - Test API endpoints before frontend
2. **Use the API documentation** - Visit http://localhost:8000/docs
3. **Try different images** - Test with various CIFAR-10 classes
4. **Check console logs** - Both browser console and terminal for errors
5. **Save your model** - Training takes time, save after training completes

---

## 📝 Next Steps

After completing basic functionality:

1. ✅ Train the model and save it
2. ✅ Make predictions on test images
3. ✅ Generate explanations
4. ✅ Improve UI/UX
5. ✅ Add more explanation methods
6. ✅ Deploy to cloud (optional)

---

## 📄 License

This is an educational project for course purposes.

---

## ❓ Questions?

Check:
1. Troubleshooting section above
2. API documentation at http://localhost:8000/docs
3. Browser console (F12) for frontend errors
4. Terminal output for backend errors

---

## ✅ Checklist Before Submission

- [ ] Backend runs without errors
- [ ] Frontend runs without errors
- [ ] Can upload images
- [ ] Can make predictions
- [ ] Can generate LIME explanations
- [ ] Can generate Grad-CAM explanations
- [ ] All components display correctly
- [ ] No CORS errors
- [ ] Model trained and saved
- [ ] Code is commented
- [ ] README is complete

---

## 🎉 Good Luck!

You now have everything needed for a complete course project on Explainable AI!

Start with the Quick Start section and enjoy building! 🚀

---

**Last Updated**: 2024
**Version**: 1.0
**Status**: Ready for Production
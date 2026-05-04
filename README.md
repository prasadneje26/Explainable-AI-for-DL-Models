# Explainable AI for Deep Learning Models

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Architecture Overview](#-architecture-overview)
- [Workflow](#-workflow)
- [Tech Stack](#-tech-stack)
- [ML Models](#-ml-models)
- [Datasets](#-datasets)
- [Training Results](#-training-results)
- [Business Value](#-business-value)
- [How to Run](#-how-to-run)
- [API Documentation](#-api-documentation)
- [Challenges & Solutions](#-challenges--solutions)
- [Future Work](#-future-work)

---

## 🎯 Problem Statement

### The Core Problem: Black Box AI

Deep learning models excel at making accurate predictions across domains like image recognition, text analysis, and data classification. However, these models operate as **"black boxes"**—they provide predictions without explaining the reasoning behind their decisions.

### Why This Matters

**Real-World Consequences:**
- **Medical Diagnosis**: AI detects cancer but can't explain which features in the scan led to the diagnosis
- **Financial Decisions**: Banks deny loans with AI recommendations but lack transparency for customers
- **Autonomous Systems**: Self-driving cars make critical decisions without explainable reasoning
- **Bias Detection**: Models may learn biased patterns from training data without visibility

**Regulatory & Trust Issues:**
- **GDPR & HIPAA**: Require explainable AI for high-stakes decisions
- **Accountability**: No way to audit or debug AI decisions
- **User Trust**: People prefer systems they can understand
- **Bias Amplification**: Hidden biases in training data get perpetuated

### The Solution: Explainable AI (XAI)

This project builds a complete **Explainable AI system** that not only makes predictions but also provides **human-understandable explanations** using **SHAP (SHapley Additive exPlanations)**—a mathematically rigorous method based on game theory.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                       │
│              (localhost:3000)                           │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ ImageModel  TextModel  TabularModel  AudioModel │   │
│  │ Components with Upload/Input UI                 │   │
│  └─────────────────────────────────────────────────┘   │
└────────────┬────────────────────────────────────────────┘
             │ HTTP/JSON API Calls
             ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI Backend                            │
│              (localhost:8000)                           │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  /predict/image                                  │   │
│  │  /predict/text                                   │   │
│  │  /predict/tabular                                │   │
│  │  /predict/audio                                  │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │ TensorFlow Models  +  SHAP Explainers            │   │
│  │ (CNN, LSTM, DNN, 1D-CNN)                         │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Component Breakdown

**Frontend Layer (React)**
- **Purpose**: User interface for data input and result display
- **Components**: ImageModel, TextModel, TabularModel, AudioModel
- **Features**: File uploads, form inputs, prediction display, explanation visualization

**API Layer (FastAPI)**
- **Purpose**: REST API endpoints for predictions and explanations
- **Endpoints**: 4 prediction routes + health check
- **Features**: Automatic OpenAPI documentation, async request handling

**ML Layer (TensorFlow + SHAP)**
- **Models**: 4 trained neural networks
- **Explainers**: SHAP classes for each model type
- **Processing**: Preprocessing, inference, explanation generation

---

## 🔄 Workflow

### End-to-End User Workflow

**Step 1: Data Input**
```
User selects model type:
├── Upload image file (JPEG/PNG)
├── Enter text (movie review, article, etc.)
├── Fill form (Iris measurements: sepal/petal length/width)
└── Upload audio file (WAV/MP3) or select signal type
```

**Step 2: Preprocessing**
```
Image: Resize to 32×32 → Normalize pixels to [0,1]
Text: Tokenize words → Convert to word IDs → Pad to 200 tokens
Tabular: Standardize features (mean=0, std=1)
Audio: Normalize amplitude to [-1, 1] → Reshape to (1000, 1)
```

**Step 3: Model Inference**
```
Load trained model → Run forward pass → Get prediction probabilities
CNN: 10 classes (airplane, car, bird, etc.)
LSTM: 2 classes (Positive, Negative sentiment)
DNN: 3 classes (Setosa, Versicolor, Virginica)
1D-CNN: 3 classes (Sine, Square, Noise)
```

**Step 4: SHAP Explanation**
```
Initialize SHAP explainer → Calculate attributions → Generate visualization
DeepExplainer: Pixel importance heatmap
GradientExplainer: Word/feature importance bars
KernelExplainer: Feature contribution bars
```

**Step 5: Result Display**
```
Show prediction + confidence + explanation visualization
Image: Original + SHAP overlay heatmap
Text: Prediction + word importance bar chart
Tabular: Prediction + feature contribution bars
Audio: Prediction + time-domain attribution plot
```

---

## 🛠️ Tech Stack

### Backend Stack (Python Ecosystem)

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Web Framework** | FastAPI | 0.104.0 | REST API with automatic docs |
| **ASGI Server** | Uvicorn | 0.24.0 | Async server for FastAPI |
| **File Handling** | python-multipart | 0.0.6 | Parse file uploads |
| **Data Validation** | Pydantic | 2.5.0 | Request/response validation |
| **Deep Learning** | TensorFlow | ≥2.13.0 | Neural network framework |
| **Explainability** | SHAP | ≥0.43.0 | SHAP explanations |
| **Visualization** | Matplotlib | ≥3.7.0 | Plot generation |
| **Data Science** | scikit-learn | ≥1.3.0 | Preprocessing, metrics |

### Frontend Stack (JavaScript Ecosystem)

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **UI Framework** | React | ^18.2.0 | Component-based UI |
| **HTTP Client** | axios | ^1.6.2 | API communication |
| **Build Tools** | react-scripts | 5.0.1 | Webpack, Babel, ESLint |

---

## 🤖 ML Models

### Model Overview

| Model | Architecture | Input Shape | Output Classes | Parameters |
|-------|--------------|-------------|----------------|------------|
| **Image CNN** | Convolutional Neural Network | (32, 32, 3) | 10 | 816,938 |
| **Text LSTM** | Bidirectional LSTM | (200,) | 2 | 1,010,370 |
| **Tabular DNN** | Dense Neural Network | (4,) | 3 | ~10,000 |
| **Audio 1D-CNN** | 1D Convolutional Network | (1000, 1) | 3 | ~50,000 |

### Image CNN (CIFAR-10 Classification)

**Architecture:**
```python
Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    # ... more conv layers ...
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

### Text LSTM (IMDB Sentiment Analysis)

**Architecture:**
```python
Sequential([
    Embedding(10000, 64, input_length=200),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])
```

### Tabular DNN (Iris Classification)

**Architecture:**
```python
Sequential([
    Dense(128, activation='relu', input_shape=(4,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(3, activation='softmax')
])
```

### Audio 1D-CNN (Signal Classification)

**Architecture:**
```python
Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(1000, 1)),
    MaxPooling1D(2),
    Dropout(0.3),
    Conv1D(64, 3, activation='relu'),
    MaxPooling1D(2),
    Dropout(0.3),
    Conv1D(128, 3, activation='relu'),
    GlobalAveragePooling1D(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])
```

---

## 📊 Datasets

| Dataset | Type | Size | Classes | Source | Purpose |
|---------|------|------|---------|--------|---------|
| **CIFAR-10** | Images | 60,000 | 10 | Keras | Object recognition |
| **IMDB** | Text | 50,000 | 2 | Keras | Sentiment analysis |
| **Iris** | Tabular | 150 | 3 | scikit-learn | Species classification |
| **Synthetic Audio** | Signals | 2,400 | 3 | Generated | Signal type detection |

### CIFAR-10 Dataset
- **60,000 color images** (32×32 pixels)
- **10 object categories**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Preprocessing**: Normalization to [0,1], one-hot encoding

### IMDB Dataset
- **50,000 movie reviews** with sentiment labels
- **Vocabulary**: 10,000 most frequent words
- **Preprocessing**: Tokenization, padding to 200 tokens, one-hot encoding

### Iris Dataset
- **150 flower samples** with 4 measurements each
- **Features**: Sepal length/width, petal length/width
- **Classes**: Setosa, Versicolor, Virginica
- **Preprocessing**: Standardization, one-hot encoding

### Synthetic Audio Dataset
- **2,400 generated waveforms** (800 each type)
- **Types**: Sine waves, square waves, random noise
- **Preprocessing**: Amplitude normalization, reshaping for 1D CNN

---

## 📈 Training Results

| Model | Dataset | Test Accuracy | Status | Training Time |
|-------|---------|---------------|--------|---------------|
| **Tabular DNN** | Iris (150 samples) | **93.33%** | ✅ Excellent | ~2 minutes |
| **Text LSTM** | IMDB (50,000 reviews) | **87.38%** | ✅ Good | ~40 minutes |
| **Image CNN** | CIFAR-10 (60,000 images) | **74.96%** | ✅ Decent | ~15 minutes |
| **Audio 1D-CNN** | Synthetic (2,400 signals) | **32.92%** | ⚠️ Overfitting | ~6 minutes |

### Key Insights
- **Tabular model**: Best performance on small, clean dataset
- **Text model**: Strong performance on sequential data
- **Image model**: Reasonable accuracy for CIFAR-10 benchmark
- **Audio model**: Overfitting detected - model memorized training patterns

---

## 💼 Business Value

### Regulatory Compliance
- **GDPR/HIPAA**: Explainable AI for high-stakes decisions
- **Audit Trail**: Transparent decision-making process
- **Bias Detection**: Identify and mitigate unfair patterns

### Operational Benefits
- **Trust Building**: Users understand and trust AI recommendations
- **Debugging**: Faster identification of model issues
- **Accountability**: Clear reasoning for AI decisions
- **Risk Reduction**: Avoid black box decision failures

### Competitive Advantages
- **Differentiation**: Explainable vs. opaque competitors
- **User Adoption**: Higher acceptance of AI systems
- **Cost Savings**: Reduced time explaining AI decisions
- **Innovation**: Foundation for responsible AI development

---

## 🚀 How to Run

### Prerequisites
- Python 3.12+
- Node.js 14+
- Git

### 1. Backend Setup

```bash
cd backend
pip install -r requirements.txt
python app.py
```

**API will be available at:** `http://localhost:8000`
**Interactive docs:** `http://localhost:8000/docs`

### 2. Frontend Setup

```bash
cd frontend
npm install
npm start
```

**App opens at:** `http://localhost:3000`

### 3. Optional: Retrain Models

```bash
cd backend
python train.py
```

This trains all 4 models and saves them as `.keras` files.

---

## 📚 API Documentation

### Endpoints

| Method | Endpoint | Input | Output |
|--------|----------|-------|--------|
| `GET` | `/health` | None | System status |
| `POST` | `/predict/image` | Image file | Prediction + SHAP explanation |
| `POST` | `/predict/text` | JSON text | Prediction + SHAP explanation |
| `POST` | `/predict/tabular` | JSON features | Prediction + SHAP explanation |
| `POST` | `/predict/audio` | Audio file | Prediction + SHAP explanation |

### Example API Call

```javascript
// Image prediction
const formData = new FormData();
formData.append('file', imageFile);

const response = await axios.post('/predict/image', formData);
console.log(response.data);
// {
//   "prediction": "cat",
//   "confidence": 0.85,
//   "explanation": "Model focused on ears and whiskers...",
//   "shap_image": "data:image/png;base64,..."
// }
```

---

## 🛠️ Challenges & Solutions

### Technical Challenges Faced

1. **Model-Data Mismatch**
   - **Problem**: Image model expected MNIST (28×28×1) but got CIFAR-10 (32×32×3)
   - **Solution**: Updated CNN architecture to handle RGB images

2. **Import Errors**
   - **Problem**: Training script referenced non-existent functions
   - **Solution**: Aligned imports with actual data.py functions

3. **Overfitting in Audio Model**
   - **Problem**: 99%+ training accuracy but 32.92% validation accuracy
   - **Solution**: Implemented early stopping and increased dropout

4. **SHAP Method Selection**
   - **Problem**: Different models require different SHAP explainers
   - **Solution**: Used DeepExplainer (CNN), GradientExplainer (LSTM/1D-CNN), KernelExplainer (DNN)

5. **Preprocessing Pipeline**
   - **Problem**: Inconsistent data formats across models
   - **Solution**: Standardized preprocessing functions for each data type

### Lessons Learned

- **Data-Model Alignment**: Ensure preprocessing matches model expectations
- **Validation Monitoring**: Track validation metrics to detect overfitting early
- **SHAP Compatibility**: Choose appropriate explainer for each model architecture
- **Error Handling**: Robust error handling for file uploads and API requests

---

## 🔮 Future Work

### Immediate Extensions
- **LIME Integration**: Alternative explanation method for comparison
- **Grad-CAM**: Additional visualization for CNN models
- **Model Comparison**: Side-by-side explanations from different models
- **Batch Processing**: Handle multiple inputs simultaneously

### Advanced Features
- **Bias Detection**: Automated fairness analysis and reporting
- **Model Monitoring**: Track explanation quality over time
- **User Feedback**: Incorporate user corrections into model updates
- **Multi-modal**: Combine multiple data types for richer explanations

### Production Considerations
- **Docker Deployment**: Containerized application for easy deployment
- **Model Versioning**: Track model versions and explanation changes
- **Scalability**: Handle higher request volumes with load balancing
- **Security**: Input validation and rate limiting for production use

---

## 📞 Support & Contact

For questions about this project:
- Review the code in `backend/` and `frontend/` directories
- Check API documentation at `http://localhost:8000/docs`
- Examine model architectures in `backend/models.py`
- Review SHAP implementations in `backend/explainers.py`

---

## 📄 License

This project is developed for educational and research purposes in Explainable AI.

---

*Last updated: April 20, 2026*
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
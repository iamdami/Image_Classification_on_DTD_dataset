# Texture Classification with EfficientNet-B0

This project performs **image classification** on a texture dataset (such as DTD) using the **EfficientNet-B0** architecture.  
It is designed to be lightweight, accurate, and suitable for assignments requiring submission size limits.

---

## Structure

<pre>
Image_Classification_on_DTD_dataset/
├── data/                # Test images (flat folder, e.g., 0001.jpg, …)
├── utils/               # DataLoader and preprocessing utilities
│
├── model.py             # EfficientNet-B0 model definition
├── train.py             # Training logic
├── train_main.py        # Entry point for model training
├── run.py               # Inference script (produces result.txt)
├── test.py              # Optional evaluation script
├── result.txt           # Final prediction output
</pre>

---

## How to Use

### 1. Run Inference (Prediction)

**1. Place test images (JPG/PNG) into the `data/` folder:**
<pre>
data/
├── 0001.jpg
├── 0002.jpg
├── …
</pre>

**2. Run the inference script:**
```bash
python run.py --load-model checkpoints/model.pth --data data/
```

**3.	Predictions will be saved to:**
```result.txt```

Each line corresponds to the predicted class index (0–19) for one image.  

### 2. Train the Model (Optional)  

To train your own model using an ImageFolder-style dataset:  
```
data/
├── 0/
│   ├── img1.jpg
├── 1/
│   ├── img2.jpg
...
```
Then run: ```python train_main.py```

The trained model will be saved to: ```checkpoints/model.pth```

---

## Results (Final Model Performance)

- Model: EfficientNet-B0
- Training Accuracy: ~99.88%
- Prediction Distribution:
 
 <pre>
   Counter({
  0: 120, 1: 120, ..., 19: 120
})
 </pre>

Uniform prediction across all 20 classes  
Final model size: 16MB (state_dict only)  

---

## Author
	•	Name: 이다미 (Dami Lee)
	•	Course: Deep Learning (Spring 2025, Hanyang Univ. ERICA)

---

## Notes
- This project uses torchvision.models.efficientnet_b0 with pretrained=True removed.
- The model is saved using torch.save(model.state_dict()) for compact submission.
- For testing, test images must be sorted and filename-cleaned (e.g., 0001.jpg, not 0_class_0001.jpg).

  

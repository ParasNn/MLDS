# MLDS — PyTorch MNIST CNN

This repository contains simple machine-learning examples used for coursework.

Contents
- `p0.py` — Example script that downloads a dataset (uses scikit-learn fetchers) and also trains a small model (scripted example of data loading, training loop, and basic evaluation). Outputs a saved model checkpoint when training completes.
- `p1.ipynb` — PyTorch notebook: builds and trains a small CNN on MNIST. Includes model definition, dataloaders, training loop, and evaluation.
- `data/` — Local dataset cache (MNIST files are under `data/MNIST/raw/`).

Quick setup (macOS, zsh)
1. Create and activate a virtual environment (recommended):

```bash
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies (adjust torch install for your CUDA/CPU choice):

```bash
pip install --upgrade pip
pip install torch torchvision matplotlib tqdm scikit-learn
```

Notes on Python version
- This project was developed with Python 3.9. Any modern 3.8+ interpreter should work, but adjust the venv command if you use a different python.

Running the notebook
- Start Jupyter in the project folder and open `p1.ipynb`:

```bash
pip install jupyterlab
jupyter lab .
```

- Or open the notebook in VS Code and run the cells interactively.

Running the script
- To run the simple script `p0.py`:

```bash
source .venv/bin/activate
python "ML Pytorch/p0.py"
```

SSL certificate troubleshooting (macOS)
- If fetching datasets fails with an SSL certificate error (CERTIFICATE_VERIFY_FAILED), try:
  1. Run the Python "Install Certificates" helper that ships with python.org installers. Example (adjust path/version):
     ```bash
     open "/Applications/Python 3.9/Install Certificates.command"
     ```
  2. Or install `certifi` and point Python to its CA bundle:
     ```bash
     pip install certifi
     export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
     python "ML Pytorch/p0.py"
     ```
  3. As a last-resort debug workaround (INSECURE), you can temporarily disable verification in code:
     ```python
     import ssl
     ssl._create_default_https_context = ssl._create_unverified_context
     ```
     Do not use this in production.

What to look for (evaluation / overfitting checks)
- Compare training and validation accuracy and loss through epochs.
- Evaluate the final model on the official MNIST test set (train=False) to confirm generalization.
- If validation >> training or validation loss increases while training loss decreases, try weight decay, dropout, or data augmentation.

Useful commands to add to the notebook (evaluation)
- Test accuracy on MNIST test split (example):
```python
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=256)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Test Accuracy: {100.*correct/total:.2f}%")
```

Extras you can add
- Early stopping and model checkpointing (save best `model.state_dict()`).
- Learning rate scheduler (e.g., `ReduceLROnPlateau`).
- Training/validation loss and accuracy plots.

License / Attribution
- This is a course/homework project. No external license specified.

If you want, I can:
- Patch `p1.ipynb` to add test-evaluation and plotting cells, or
- Create a `requirements.txt` listing exact packages and versions.


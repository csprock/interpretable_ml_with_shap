# Interpretable Machine Learning with SHAP Values


This repository contains the code for the presentation ["Interpretable Machine Learning with SHAP Values"](https://genevalytics.github.io/events/interpretable-ml-with-shap/) given at the University of Geneva on April 24th, 2025 as part of the [Genevalytics](https://genevalytics.github.io/) speaker series. 

Explaining how a model arrived at a prediction is an important part of using machine learning. In this hands-on presentation, we will be exploring how to use SHAP values, a widely used model interpretation method, and compare them with other methods. You will learn what SHAP values are and how to apply them using the popular shap Python library.

## Repository Content

This repository includes:

```
interpretable_ml_with_shap/
├── demo_notebook.ipynb   # presentation notebook
├── rf_best_params.json   # model parameters
├── california_housing_descr.txt   # description of the California housing dataset
├── demo_notebook.ipynb   # presentation notebook
├── .devcontainer/        # Development container configuration
├── requirements.txt      # Python dependencies for development container
└── README.md             # This file
```

## Getting Started

This project can be set up quickly using development containers, which provide a consistent and isolated environment for development.

### With Docker

#### Prerequisites

- [Docker](https://www.docker.com/get-started)
- [Visual Studio Code](https://code.visualstudio.com/)
- [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) for VS Code

#### Setup Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/csprock/interpretable_ml_with_shap.git
    cd interpretable_ml_with_shap
    ```

2. Open the project in VS Code:
    ```bash
    code .
    ```

3. When prompted to "Reopen in Container", click "Yes". Alternatively, you can:
    - Open the Command Palette (F1 or Ctrl+Shift+P)
    - Select "Remote-Containers: Reopen in Container"

4. Wait for the container to build. This might take a few minutes the first time.

5. Once inside the container, the environment is ready with all necessary dependencies installed.

### Without Docker

#### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

#### Setup Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/csprock/interpretable_ml_with_shap.git
    cd interpretable_ml_with_shap
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    
    # On Windows
    venv\Scripts\activate
    
    # On macOS/Linux
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

5. Open `demo_notebook.ipynb` in your browser to run the presentation notebook.
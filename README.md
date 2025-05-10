# Oil Analysis

This repository provides a comprehensive analysis of lubricant oil wear particles using synthetic and real-world datasets. The project encompasses data preprocessing, exploratory data analysis, and predictive modeling to understand wear patterns and forecast future values.

## Table of Contents

* [Overview](#overview)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Datasets](#datasets)
* [Results](#results)
* [Contributing](#contributing)
* [License](#license)

## Overview

The primary objective of this project is to analyze wear particles in lubricant oils to predict equipment health and maintenance needs. By leveraging statistical analysis and machine learning techniques, the project aims to provide insights into wear trends and potential future failures.

## Project Structure

The repository is organized as follows:

* `lubricant_analysis.ipynb`: Exploratory data analysis and visualization of wear particle data.
* `lubricant-exponential-increase.ipynb`: Modeling and prediction of wear particle trends over time.
* `preprocess.py`: Script for preprocessing raw datasets, including cleaning and feature engineering.
* `wear_particle_data.csv`: Real-world dataset containing wear particle measurements.
* `synthetic_particle_data_convex.csv`: Synthetic dataset generated for modeling purposes.
* `final_particle_data.xlsx`: Processed dataset used for final analysis and modeling.
* `Report on future values prediction.pdf`: Detailed report on predictive modeling outcomes and interpretations.
* `requirements.txt`: List of Python dependencies required to run the project.([GitHub Docs][1])

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/chinu0609/oil_analysis.git
   cd oil_analysis
   ```



2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```



3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```



## Usage

1. **Preprocess the data:**

   ```bash
   python preprocess.py
   ```



This script will clean the raw datasets and generate processed files for analysis.

2. **Run the Jupyter notebooks:**

   ```bash
   jupyter notebook
   ```



Open and execute the following notebooks in order:

* `lubricant_analysis.ipynb`
* `lubricant-exponential-increase.ipynb`

These notebooks contain the exploratory analysis and predictive modeling steps.

## Datasets

* **wear\_particle\_data.csv**: Contains real-world measurements of wear particles in lubricant oils, including features such as particle size, count, and sampling time.
* **synthetic\_particle\_data\_convex.csv**: A synthetically generated dataset to simulate wear particle behavior under controlled conditions.
* **final\_particle\_data.xlsx**: The final processed dataset combining relevant features for modeling purposes.

## Results

The analysis revealed significant patterns in wear particle accumulation over time. Predictive models demonstrated the capability to forecast future wear trends, aiding in proactive maintenance scheduling. Detailed findings and visualizations are documented in the [Report on future values prediction.pdf](Report%20on%20future%20values%20prediction.pdf).

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements. For major changes, kindly open an issue first to discuss proposed modifications.

## License

This project is licensed under the [MIT License](LICENSE).


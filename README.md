# kaggle-jane-street-forecasting
## Getting Started
1. Install required packages using `pip install -r requirements.txt`. It is recommended to use Conda or other virtual environment.
1. Setup the Kaggle API following the [official instructions](https://github.com/Kaggle/kaggle-api?tab=readme-ov-file#installation).
2. Create the data directory and download the dataset
```
mkdir -p /kaggle/input \
 && kaggle competitions download -c jane-street-real-time-market-data-forecasting \
 && mkdir jane-street-real-time-market-data-forecasting \
 && unzip jane-street-real-time-market-data-forecasting.zip -d jane-street-real-time-market-data-forecasting \
 && mv jane-street-real-time-market-data-forecasting /kaggle/input/
```
3. To train a model, run `python train/train.py`
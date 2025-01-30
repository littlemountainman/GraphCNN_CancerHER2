## Graph Convolutional Neural Network for HER2 Cancer inhibitor search

### Installing Requirements:
```
pip3 install -r requirements.txt
```

### Unpacking the data in /data/preprocessed/

DO COPY the content of the zip into /data/preprocessed

### Usage:

* Model training happens on KIBA, for the data generation, please check this repo:(https://github.com/thinng/GraphDTA/tree/master)
```
python3 train.py
```
* Prediction on BindingDB (https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp) to find relvant drug candidates
```
python3 predict_bdb.py
```

This should output a file within the results/ folder which can be further examined.



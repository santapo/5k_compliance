## Zalo AI challenge: 5K Compliance

## Install dependencies
```console
conda create -n 5k_compliance python=3.7
conda activate 5k_compliance
pip install -r requirements.txt
```
## File and Directory
`face_detector`: contains face detector model and its weights  
`mask_classifier`: contains mask classifier weights  
`detect_mask_image.py`: inference code of face mask detection  
`5k_compliance.py`: run face mask detector in a list of images and classify whether or not these images is 5k compliance. After that save the result to `submit_public_test.csv`
## Running

```python
python 5k_compliance.py -i data/public_test/images
```

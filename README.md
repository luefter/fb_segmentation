# fb_segmentation
Package for applying forground-background segmentation

# Usage
python -m pip install -e .[dev]

### Sources
## Code Sources
https://github.com/milesial/Pytorch-UNet


###
Trimaps have values 1,2,3. Whereas 2 stands for background
convert to 0,1 by (trimap != 2).astype(np.uint8)


## USAGE OF INFERENCE.PY
python src/fbs/inference.py --img-name Abyssinian_2.jpg --save True


## USAGE OF INFERENCE.PY
python src/fbs/inference.py --img-name Abyssinian_2.jpg --save True

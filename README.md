# incucyte_stabilizer

Takes a directory of TIFF movies exported from an Incucyte microscope and
applies image stabilization to each one. If multiple channels are present,
they will be merged into one OME-TIFF movie for each well and field.

```
usage: python stabilize.py [-h] -i INPUT_PATH -o OUTPUT_PATH [-n NUM_WORKERS] [-v]

options:
  -h, --help            show this help message and exit
  -i, --input-path INPUT_PATH
  -o, --output-path OUTPUT_PATH
  -v, --verbose
```

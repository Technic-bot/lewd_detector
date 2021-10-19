# Nudenet Wrapper. 

Small warpper script for [NudeNet](https://github.com/notAI-tech/NudeNet) Made to provide a small and easy to use cli interface with the library.

You can indicate a single image from command line and Wrapper script will run in through NudeNet and output both Classification score and the image labeled with detections via bounding boxes. It also provides an option to output labeled image to disk:

Example usage 
```bash
python3  lewd_detector.py  --probability 0.15  --mode base images/image.png --outfile results/image.png
```



from nudenet import NudeDetector
from nudenet import NudeClassifier

import cv2

import argparse

from collections import deque 

import pprint


def proc_opts():
  parser = argparse.ArgumentParser('Lewd detector (NudeNet based)')
  parser.add_argument('image', default='Image file to read')
  parser.add_argument('--noviz', default=False, action='store_true')
  parser.add_argument('--outfile',default=None)
  parser.add_argument('--mode',default='base')
  parser.add_argument('--probability', type=float, default=0.1, help='Min probability to consider')
  return parser.parse_args()

class nudeWrapper():
  def __init__(self):
  
    self.detector = NudeDetector() 
    self.classifier = NudeClassifier()
    return 

  def detect_sketch(self,filename, prob=0.1,mode='base'):
    """ We do the detection here"""
    results = self.detector.detect(filename,mode=mode,min_prob=prob)
    nms_results = self.nms(results)
    return nms_results

  def classify_sketch(self,filename):
    return  self.classifier.classify(filename)
   
  def nms(self,boxes,l=0.4):
    """Basic non maximun supression algorithm"""
    n_boxes = len(boxes)
    keep = []
    # O(n2)
    while boxes:
      box = boxes.pop()
      add_box = True
      for another_box in boxes:
        #if box['label'] == another_box['label']:
        iou_val = self.iou(another_box['box'],box['box'])
        if iou_val > l:
          add_box = False
          break
      if add_box:
        keep.append(box)
            
        
    print("Pruned {} boxes".format(n_boxes-len(keep)))
    return keep 

  def iou(self,box1,box2):
    """intersection over union"""
    # box = [x_1, y_1 , x_2, y_2 ]
    x1 = max(box1[0],box2[0])
    y1 = max(box1[1],box2[1])
    
    x2 = min(box1[2],box2[2])
    y2 = min(box1[3],box2[3])

    interArea = max(0,x2-x1+1) * max(0,y2-y1+1)
    
    box1Area = (box1[2] - box1[0])*(box1[3] - box1[1]) 
    box2Area = (box2[2] - box2[0])*(box2[3] - box2[1]) 

    iou = interArea / (box1Area + box2Area - interArea)

    return iou
  
  def put_bounding_box(self,image_file,boxes,unsafe_prob,threshold,
                      viz=True,outfile=None):
    """We need other lib for visualization and processing, using cv2"""
    img = cv2.imread(image_file) 
    for box in boxes:
      print("Detection {} with probability {}".format(box['label'],box['score']))
      caption = box['label'].capitalize()# + "@" + str(box['score'])
      start_corner = (box['box'][0:2])
      end_corner = (box['box'][2:4])
      cv2.putText(img,caption,
          start_corner,
          cv2.FONT_HERSHEY_COMPLEX,
          1,
          (247,153,29 ),
          2)
      cv2.rectangle(img,start_corner,end_corner,(114,205,238),2)

    title="NSFW score: {:0.4}".format(unsafe_prob)
    cv2.putText(img,title,
        (0,30),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (247,153,29 ),
        2)
    
    txt = "Threshold: {:0.4}".format(threshold)
    cv2.putText(img,txt,
        (0,65),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (247,153,29 ),
        2)
    

    if viz:
      source_window = 'Lewdness detector'
      cv2.namedWindow(source_window)
      cv2.imshow(source_window, img)
      cv2.waitKey()
  
    if outfile:
      cv2.imwrite(outfile,img)

if __name__=="__main__":
  args = proc_opts()

  lwd = nudeWrapper()
  boxes = lwd.detect_sketch(args.image,args.probability,args.mode)
  classification = lwd.classify_sketch(args.image)

  print(classification)
  pckg = classification[args.image]
  lwd.put_bounding_box(args.image,boxes,pckg['unsafe'],
      args.probability,
      viz=not args.noviz,outfile=args.outfile)



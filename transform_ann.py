import os
import glob
import  argparse
import numpy as np
import csv

def parse_args():
  parser = argparse.ArgumentParser(description='transform dataset')
  parser.add_argument('--data_path', type=str, help='image_path')
  parser.add_argument('--save_path', type=str, help='generate anno')
  args = parser.parse_args()
  return args

args = parse_args()

image_path = args.data_path
# image_path = '/home/zhao/work/dataset/icdar2015'
files_list = []
for ext in ['jpg', 'png', 'jpeg', 'JPG','png', 'PNG']:
  files_list.extend(glob.glob(os.path.join(image_path, '*.{}'.format(ext))))

lines = []

for item in files_list:
  imae_path = item
  txt_fn = item[:-3] + 'txt'
  with open(txt_fn, 'r') as f:
    reader = csv.reader(f)
    ann = ''
    for line in reader:
      label = line[-1]
      line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
      if label == '*' or label == '###':
        text_tags = 1
      else:
        text_tags = 0
      ann = ','.join([ann, *line[:8], str(text_tags)])
    lines.append(imae_path+ann)


with open(args.save_path, 'w') as f:
  for line in lines:
    f.writelines(line+'\n')
import argparse, os, os.path, glob, random, sys, json
from collections import defaultdict
from lxml import objectify

from scipy.misc import imread, imsave, imresize
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

train_anns_path = '/share/data/vision-greg/ImageNetbbox/clsloc/train'
train_image_dir = '/share/data/vision-greg/ImageNet_flat/clsloc/images/train'
val_anns_path = '/share/data/vision-greg/ImageNetbbox/clsloc/val'
val_image_dir = '/share/data/vision-greg/ImageNet_flat/clsloc/images/val'


def get_synset_stats():
  with open('./words.txt') as f:
    wnid_to_words = dict(line.strip().split('\t') for line in f)

  wnids = os.listdir(train_anns_path)
  wnid_to_stats = {wnid: {} for wnid in wnids}
  for i, wnid in enumerate(wnids):
    synset_dir = os.path.join(train_anns_path, wnid)
    bbox_files = os.listdir(synset_dir)
    bbox_files = [os.path.join(synset_dir, x) for x in bbox_files]

    glob_str = '%s_*.JPEG' % wnid
    img_files = glob.glob(os.path.join(train_image_dir, glob_str))

    wnid_to_stats[wnid]['bbox_files'] = bbox_files
    wnid_to_stats[wnid]['img_files'] = img_files
    wnid_to_stats[wnid]['num_imgs_train'] = len(img_files)
    wnid_to_stats[wnid]['num_loc_train'] = len(bbox_files)
    wnid_to_stats[wnid]['words'] = wnid_to_words[wnid]

    print(i, file=sys.stderr)
    print('%d\t%s\t%s\t%d\t%d' % (
        i, wnid, wnid_to_words[wnid], len(bbox_files), len(img_files)))

    
def parse_xml_file(filename):
  with open(filename, 'r') as f:
    xml = f.read()
  ann = objectify.fromstring(xml)
  if ann.filename != '%s':
    img_filename = '%s.JPEG' % ann.filename
  else:
    img_filename = filename[filename.rfind('/')+1:-3] + 'JPEG'
  bbox = ann.object.bndbox
  bbox = [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]
  bbox = [int(x) for x in bbox]
  name = str(ann.object.name)
  return img_filename, bbox, name


def resize_image(img, size, bbox=None, crop=True, show=False):
  """
  Resize an image and its bounding box to a square.

  img - A numpy array with pixel data for the image to resize.
  size - Integer giving the height and width of the resized image.
  bbox - Optionally, a list [xmin, ymin, xmax, ymax] giving the coordinates
         of a bounding box in the original image.
  crop - If true, center crop the original image before resizing; this avoids
         distortion in images with nonunit aspect ratio, but may also crop out
         part of the object.
  show - If true, show the original and resized image and bounding box.

  Returns:
  If bbox was passed: (img_resized, bbox_resized)
  otherwise: img_resized
  """

  def draw_rect(coords):
    width = coords[2] - coords[0]
    height = coords[3] - coords[1]
    rect = Rectangle((coords[0], coords[1]), width, height, 
                     fill=False, linewidth=2.0, ec='green')
    plt.gca().add_patch(rect)

  img_resized = img
  if bbox is not None:
    bbox_resized = [x for x in bbox]
  if crop:
    h, w = img.shape[0], img.shape[1]
    if h > w:
      h0 = (h - w) // 2
      if bbox is not None:
        bbox_resized[1] -= h0
        bbox_resized[3] -= h0
      img_resized = img[h0:h0+w, :]
    elif w > h:
      w0 = (w - h) // 2
      if bbox is not None:
        bbox_resized[0] -= w0
        bbox_resized[2] -= w0
      img_resized = img[:, w0:w0+h]

  if bbox is not None:
    h_ratio = float(size) / img_resized.shape[0]
    w_ratio = float(size) / img_resized.shape[1]
    ratios = [w_ratio, h_ratio, w_ratio, h_ratio]
    bbox_resized = [int(1 + r * (x - 1)) for x, r in zip(bbox_resized, ratios)]
    bbox_resized = np.clip(bbox_resized, 0, size - 1)
  img_resized = imresize(img_resized, (size, size))

  if show:
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    if bbox is not None:
      draw_rect(bbox)
    plt.subplot(1, 2, 2)
    plt.imshow(img_resized)
    if bbox is not None:
      draw_rect(bbox_resized)
    plt.show()

  if bbox is None:
    return img_resized
  else:
    return img_resized, bbox_resized


def write_data_in_synset_folders(part_data, part, out_dir, image_size):
  part_dir = os.path.join(out_dir, part)
  os.mkdir(part_dir)
  num_wnids = len(part_data)
  for i, (wnid, wnid_data) in enumerate(part_data.items()):
    print('Writing images for synset %d / %d of %s' % (i + 1, num_wnids, part))
    wnid_dir = os.path.join(part_dir, wnid)
    os.mkdir(wnid_dir)
    image_dir = os.path.join(wnid_dir, 'images')
    os.mkdir(image_dir)
    boxes_filename = os.path.join(wnid_dir, '%s_boxes.txt' % wnid)
    boxes_file = open(boxes_filename, 'w')
    for i, (img_filename, bbox) in enumerate(wnid_data):
      out_img_filename = '%s_%d.JPEG' % (wnid, i)
      full_out_img_filename = os.path.join(image_dir, out_img_filename)
      try: img = imread(img_filename[:58+9] + '/' + img_filename[58:], mode='RGB')
      except: img = imread(img_filename, mode='RGB')

      img_resized, bbox_resized = resize_image(img, image_size, bbox)

      #if img_resized.mode != "RGB": img_resized = img_resized.convert(mode="RGB")
      #if img_resized.shape != (64,64,3):
      #  if img_resized.shape == (64,64):
      #    img_resized = np.array([img_resized, img_resized, img_resized]).transpose((1, 2, 0))
      #  elif img_resized.shape == (64,64,4):
      #    print(img_resized)

      imsave(full_out_img_filename, img_resized)
      boxes_file.write('%s\t%d\t%d\t%d\t%d\n' % (out_img_filename,
                       bbox_resized[0], bbox_resized[1], bbox_resized[2], bbox_resized[3]))
    boxes_file.close()


def write_data_in_one_folder(part_data, part, out_dir, image_size):
  part_dir = os.path.join(out_dir, part)
  os.mkdir(part_dir)

  # First flatten the part data so we can shuffle it
  part_data_flat = []
  for wnid, wnid_data in part_data.items():
    for (img_filename, bbox) in wnid_data:
      part_data_flat.append((wnid, img_filename, bbox))

  random.shuffle(part_data_flat)
  image_dir = os.path.join(part_dir, 'images')
  os.mkdir(image_dir)

  annotations_filename = os.path.join(part_dir, '%s_annotations.txt' % part)
  annotations_file = open(annotations_filename, 'w')
  for i, (wnid, img_filename, bbox) in enumerate(part_data_flat):
    if i % 100 == 0:
      print('Finished writing %d / %d %s images' % (i, len(part_data_flat), part))
    out_img_filename = '%s_%s.JPEG' % (part, i)
    full_out_img_filename = os.path.join(image_dir, out_img_filename)
    try: img = imread(img_filename[:58+9] + '/' + img_filename[58:], mode='RGB')
    except: img = imread(img_filename, mode='RGB')

    img_resized, bbox_resized = resize_image(img, image_size, bbox)

    imsave(full_out_img_filename, img_resized)
    annotations_file.write('%s\t%s\t%d\t%d\t%d\t%d\n' % (
        out_img_filename, wnid,
        bbox_resized[0], bbox_resized[1], bbox_resized[2], bbox_resized[3]))
  annotations_file.close()


def make_tiny_imagenet(wnids, num_train, num_val, out_dir, image_size=50, test=False):
  if os.path.isdir(out_dir):
    print('Output directory already exists')
    return

  # dataset['train']['n123'][0] = (filename, (xmin, ymin, xmax, xmax))
  # gives one example of an image and bbox for synset n123 of the training subset
  dataset = defaultdict(lambda: defaultdict(list))
  for i, wnid in enumerate(wnids):
    print('Choosing train and val images for synset %d / %d' % (i + 1, len(wnids)))

    # TinyImagenet train and val images come from ILSVRC-2012 train images
    train_synset_dir = os.path.join(train_anns_path, wnid)
    orig_train_bbox_files = os.listdir(train_synset_dir)
    orig_train_bbox_files = {os.path.join(train_synset_dir, x) for x in orig_train_bbox_files}

    train_bbox_files = random.sample(orig_train_bbox_files, min(num_train, len(orig_train_bbox_files)))
    orig_train_bbox_files -= set(train_bbox_files)
    val_bbox_files = random.sample(orig_train_bbox_files, min(num_val, len(orig_train_bbox_files)))

    for bbox_file in train_bbox_files:
      img_filename, bbox, _ =  parse_xml_file(bbox_file)
      img_filename = os.path.join(train_image_dir, img_filename)
      dataset['train'][wnid].append((img_filename, bbox))

    for bbox_file in val_bbox_files:
      img_filename, bbox, _ = parse_xml_file(bbox_file)
      img_filename = os.path.join(train_image_dir, img_filename)
      dataset['val'][wnid].append((img_filename, bbox))
    
  # All the validation XML files are all mixed up in one folder, so we need to
  # iterate over all of them. Since this takes forever, guard it behind a flag.
  # The name field of the validation XML files gives the synset of that image.
  if test:
    val_xml_files = os.listdir(val_anns_path)
    for i, val_xml_file in enumerate(val_xml_files):
      if i % 200 == 0:
        print('Processed %d / %d val xml files so far' % (i, len(val_xml_files)))
      val_xml_file = os.path.join(val_anns_path, val_xml_file)
      img_filename, bbox, wnid = parse_xml_file(val_xml_file)
      if wnid in wnids:
        img_filename = os.path.join(val_image_dir, img_filename)
        dataset['test'][wnid].append((img_filename, bbox))

  # Now that we have selected the images for the dataset, we need to actually
  # create it on disk
  os.mkdir(out_dir)
  write_data_in_synset_folders(dataset['train'], 'train', out_dir, image_size)
  write_data_in_one_folder(dataset['val'], 'val', out_dir, image_size)
  write_data_in_one_folder(dataset['test'], 'test', out_dir, image_size)


parser = argparse.ArgumentParser()
parser.add_argument('--wnid_file', type=argparse.FileType('r'))
parser.add_argument('--num_train', type=int, default=500)
parser.add_argument('--num_val', type=int, default=50)
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--out_dir')
args = parser.parse_args()

if __name__ == '__main__':
  wnids = [line.strip() for line in args.wnid_file]
  print(len(wnids))
  # wnids = ['n02108089', 'n09428293', 'n02113799']
  make_tiny_imagenet(wnids, args.num_train, args.num_val, args.out_dir, 
                     image_size=args.image_size, test=True)
  sys.exit(0)

  train_synsets = os.listdir(train_anns_path)

  get_synset_stats()
  sys.exit(0)

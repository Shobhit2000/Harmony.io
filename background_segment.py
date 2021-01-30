import os
import cv2
import tarfile
import numpy as np
from PIL import Image
import tensorflow as tf

gpus = tf.config.experimental.get_visible_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu,True)

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.compat.v1.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.compat.v1.import_graph_def(graph_def, name='')

    self.sess = tf.compat.v1.Session(graph=self.graph)

  def run(self, image):

    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map

def create_pascal_label_colormap():

  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap

def label_to_color_image(label):
 
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

################################### MODEL #############################################
LABEL_NAMES = np.asarray([
  'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
  'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
  'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

MODEL = DeepLabModel('deeplab_model.tar.gz')
print('model loaded successfully!')

print(MODEL)

def segment(path, frame):
  back1 = cv2.imread(path)
  back1 = cv2.resize(back1, (600, 500))

  cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  pil_im = Image.fromarray(cv2_im)
  resized_im, seg_map = MODEL.run(pil_im)

  seg_image = label_to_color_image(seg_map).astype(np.uint8)

  seg_image = seg_image[:, :, 0] > 100
  seg_image = np.uint8(seg_image)*255
  seg_image = cv2.cvtColor(seg_image, cv2.COLOR_GRAY2BGR)
  seg_image = 255 - seg_image
  seg_image = cv2.resize(seg_image, (600, 500))

  back2 = np.copy(back1)

  masked_image = cv2.resize(frame, (600, 500))
  masked_image[seg_image != 0] = 0

  back2[seg_image == 0] = 0

  final = masked_image+back2
  return final

import numpy as np
from torchvision import transforms

def identity(M):
  return transforms.RandomAffine(degrees=0)

def rotate(M):
  return transforms.RandomRotation(M)

def translate_x(M):
  return transforms.RandomAffine(degrees=0,translate=[M,0])

def translate_y(M):
  return transforms.RandomAffine(degrees=0,translate=[0,M])

def shear_x(M):
  return transforms.RandomAffine(degrees=0,shear=[-M,M,0.,0.])

def shear_y(M):
  return transforms.RandomAffine(degrees=0,shear=[0.,0.,-M,M])

def color(M):
  return transforms.ColorJitter(saturation=M)

def autoContrast(M):
  return transforms.RandomAutocontrast(p=1)

def contrast(M):
  return transforms.ColorJitter(contrast=M)

def brightness(M):
  return transforms.ColorJitter(brightness=M)

def equalize(M):
  return transforms.RandomEqualize(p=1)

def solarize(M):
  return transforms.RandomSolarize(M,p=1)

def posterize(M):
  return transforms.RandomPosterize(M,p=1)

def sharpness(M):
  return transforms.RandomAdjustSharpness(M,p=1)

class RandAugment():
  """Custom RandAugment for pytorch"""
  def __init__(self,M,N,max_M=20):
    """
    transforms is a list of tuples in the form:
    (transform function, min distortion, max distortion)
    If the transform has a probability, it is fixed to 1
    Parameters:
      -M: positive int in range [0:max_M]
        maximum magnitude of transformations
      -N: positive int,
        average number of transformations selected
      -max_M: positive int, maximum magnitude, default:20
    """
    self.transforms = [
      (identity,0.,0.),
      (rotate,0.,180.),
      (translate_x,0.,0.7),
      (translate_y,0.,0.7),
      (shear_x,0.,60.),
      (shear_y,0.,60.),
      (autoContrast,0.,0.),
      (contrast,0.,0.9),
      (brightness,0.,0.9),
      (equalize,0.,0.),
      (solarize,255.,155.),
      (posterize,0.,8.),
      (sharpness,0.,5.),
      (color,0.,1.),
    ]
    self.M=float(M)
    self.N=N
    self.max_M=float(max_M)

  def get_transform(self, opTuple):
    """Adapt M to the max and min values of the transform"""
    op, opMin, opMax = opTuple
    M = self.M/self.max_M*(opMax-opMin)+opMin
    return op(M)

  def __call__(self):
    """Generate a set of distortions.
    Args:
    N: Number of augmentation transformations to
    apply sequentially.
    M: Magnitude for all the transformations.
    """
    sampled_ops = np.random.choice(range(len(self.transforms)), self.N)
    return [self.get_transform(self.transforms[nop]) for nop in sampled_ops]


if __name__=="__main__":
  aug=RandAugment()
  print(list(aug(4,2)))





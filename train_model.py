# Import the functions, classes, and variables from fastai's vision module
from fastai.vision.all import *

# Get the canonicalized url pet pictures. The untrar_data function already knows where to find these pet pictures.
path = untar_data(URLs.PETS)

# Load the pet images in the desired format to be used in the data set.
dls = ImageDataLoaders.from_name_re(path, get_image_files(path/'images'), pat='(.+)_\d+.jpg', item_tfms=Resize(460), batch_tfms=aug_transforms(size=224, min_scale=0.75))

# Train the resnet50 model with the created dataset.
learn = vision_learner(dls, models.resnet50, metrics=accuracy)
learn.fine_tune(1)

# Export the trained model.
learn.path = Path('.')
learn.export()
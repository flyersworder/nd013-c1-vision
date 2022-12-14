# Object Detection in an Urban Environment

## Project overview
This is project is the final project of Udacity Data Science Nanodegree where we practice ML for object detection for a self driving car system. This is a pivotal application given the safety and ethical perspectives of a self-driving car. Through this exercise, I also learn that it is quite a daunting task that requires a lot of hard work. Luckily, we don't need to start from scratch and can rely on some previous models. And it can be more convenient with the ever-increasing computing power.

## Set up

It is a bit tricky to set the whole thing up in a Mac M1 laptop, but it is still doable following these steps (first you will need the [miniconda](https://docs.conda.io/en/latest/miniconda.html) for Mac M1):

- Follow [this page](https://developer.apple.com/metal/tensorflow-plugin/) to install the tensorflow library
- Follow [this page](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-object-detection-api-installation) to install the object detection API. After that you also need to export these folders to your pythonpath, e.g., if you use zsh, you can add the following statement to your .zshrc file:

```
export PYTHONPATH=$PYTHONPATH:tensorflow/models/
export PYTHONPATH=$PYTHONPATH:tensorflow/models/research

```

- Follow [this answer](https://stackoverflow.com/questions/70277737/cant-install-tensorflow-io-on-m1) to install `tensorflow_io`

After all these steps, you are good to go.

## Dataset

### Data

The data you will use for training, validation and testing is organized as follow:
```
data/
    - train: contain the train data (empty to start)
    - val: contain the val data (empty to start)
    - test: contains 3 files to test your model and create inference videos
```

These files have been downsampled: we have selected one every 10 frames from 10 fps videos. The `testing` folder contains frames from the 10 fps video without downsampling.

The `create_splits.py` file provides one way to split a bunch of files into these folders.

### Experiments

The experiments folder will be organized as follow:
```
experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file
    - augmentation/ - add augmentation
    - aug_batch/ - add augmentation and modify batch size
    - aug_batch_lr/ - add augmentation and modify both batch size and learning rate
    - label_map.pbtxt
```

### EDA
In [this EDA notebook](https://github.com/flyersworder/nd013-c1-vision/blob/main/Exploratory%20Data%20Analysis.ipynb), we can clearly see that we have images covering different weather and road conditions in different times of a day. Exploring 10,000 random images, we find that the data is dominated by cars and only has a small sample of pedestrians and cyclist, which are normally smaller objects than those of cars. This can cause some problems when training the model, as we can see later that the metrics (precision and recall) for medium and small objects are much worse than the large ones.

### Cross validation
In the class, we talked about cross-validation and the importance of creating meaningful training and validation splits. For this project, the `create_splits.py` file does the following:
* create three subfolders: `data/train/`, `data/val/`, and `data/test/`
* split the tf records files between these three folders, essentially using the numpy split function:
`train_files, val_file, test_file = np.split(data, [int(.75*len(data)), int(.9*len(data))])`

Use the following command to run the script:
```
python create_splits.py --data-dir data
```

## Training
You will now launch your very first experiment with the Tensorflow object detection API. Move the `pipeline_new.config` to the `experiments/reference` folder. Now launch the training process:
* a training process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
Once the training is finished, launch the evaluation process:
* an evaluation process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```

To monitor the training, you can launch a tensorboard instance by running `python -m tensorboard.main --logdir experiments/reference/`. 

Note that I only train 2,500 steps (rather than 25,000 steps) because it takes a long time to train for so many steps even in a powerful Mac M1 computer where per step time is around 0.5s. This will certainly yield a suboptimal performance. In the later stage I will need to retrain the best performing model with 25,000 steps.

### Reference experiment
The performance of this reference model, as expected, is not splendid. As it is shown below, the classification loss seems to reach a plateau already after 1,000 steps and then fluctuates around 0.7. The other losses also seem to tamper off after 2,000 steps. The precision (shown in the table in the next session) is also quite moderate: the largest precision is merely 0.21% for large objects.

![reference model](/images/model_reference.png)

### Improve on the reference
I try out three ways in order to improve the model performance

1. Augmentation. Because the smaller objects are hugely undersampled, the first augmentation method that comes to my mind is to resize the image so that we can somehow enlarge these small object. For this purpose I use `ssd_random_crop_fixed_aspect_ratio` and set the aspect ratio to 1.2. I also notice that these are mainly color photos, and change the grayscale may also help boost the performance. I thus also add this `random_rgb_to_gray` and set the probability to 0.2. I explore different kinds of augmentations using [this notebook](https://github.com/flyersworder/nd013-c1-vision/blob/main/Explore%20augmentations.ipynb).
2. Batch size. On top of the augmentation, I just slightly increase the batch size from 2 to 4.
3. Learning rate. On top of the previous two methods, I slightly decrease the learning rate from 0.013 to 0.001

The performance of these experiments (in terms of precision) are shown in the table below. As we can observe (as we purely focus on precision here), augmentation alone doesn't really improve precision (but it does improve recall), whereas augmentation plus a slightly bigger batch size improves the precision for medium and small objects. Lastly, reducing the learning rate to 0.001 also improves the performance. I also show the loss values for this experiment below.

| Metrics (Precision)                    | Reference | Augmentation | Aug + Batch | Aug + Batch + LR |
| -------------------------------------- | --------- | ------------ | ----------- | ---------------- |
| DetectionBoxes\_Precision/mAP          | 0.01%     | 0.00%        | 0.26%       | 0.20%            |
| DetectionBoxes\_Precision/mAP (large)  | 0.21%     | 0.10%        | 0.14%       | 0.08%            |
| DetectionBoxes\_Precision/mAP (medium) | 0.00%     | 0.00%        | 0.55%       | 1.35%            |
| DetectionBoxes\_Precision/mAP (small)  | 0.00%     | 0.00%        | 0.07%       | 0.23%            |
| DetectionBoxes\_Precision/mAP@.50IOU   | 0.04%     | 0.03%        | 0.71%       | 0.62%            |
| DetectionBoxes\_Precision/mAP@.75IOU   | 0.01%     | 0.00%        | 0.04%       | 0.08%            |

I realize that with this tiny precisions, these models will not be able to perform well. In fact, I already notice from the TensorBoard that the evaluation step hardly manages to identify any objects in the images. I have seen that some students are able to achieve a relatively good performance with merely 2,500 steps, but for, I need to retrain the last model in the column with 25,000 steps in order to bring the losses down to a reasonable level (see figure below).

![final model](/images/model_aug_batch_lr_25k_steps.png)


## Creating an animation
### Export the trained model

Use the function below to generate the model:

```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/aug_batch_lr/pipeline_new.config --trained_checkpoint_dir experiments/aug_batch_lr/ --output_directory experiments/aug_batch_lr/exported/
```

This should create a new folder `experiments/aug_batch_lr/exported/saved_model`. More about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).

Finally, we can create the animation using the following function:
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/aug_batch_lr/exported/saved_model --tf_record_path data/test/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/aug_batch_lr/pipeline_new.config --output_path images/animation.gif
```

### Animation
The animation is displayed below. Interestingly, the STOP sign can be misrecognized as targets.

![animation](/images/animation.gif)

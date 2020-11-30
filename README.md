# IVR assignment

Authors: Jingwen Pan (s1827818), Evripidis Papaevripidis (s1754795)

## Initiation

All of the following commands are to be run from the catkin_ws directory. Start the simulation.
```
roslaunch ivr_assignment spawn.launch
```

## 2.1

Run the angles estimator node. It publishes angles estimates to topics "joint{2/3/4}_angle_estimate".
```
rosrun ivr_assignment angles_estimator.py
```

Run the joints mover to move joints according to isntructions.
```
rosrun ivr_assignment joints_mover.py 2.1
```

## 2.2

Run target tracker. Publishes data to topics "target_{x/y/z}_estimate" and "obstacle_{x/y/z}_estimate". Requires file mgc_model.npz in project directory.
```
rosrun ivr_assignment target_tracker.py
```

## 3.1
Run end effector tracker. Publishes end effector position estimate (calculated using vision) to topics "/end_effector_position_estimate_{x/y/z}".
```
rosrun ivr_assignment end_effector_tracker.py
```

Run the angles estimator node as in part 2.1.
```
rosrun ivr_assignment angles_estimator.py
```

Run FK test. Parameter t should be a number in \[0,9\] denoting the test number to be ran. Test angles are pre-calculated. The position of the end effector
estimated using Forward Kinematics is printed. To run another test, press Ctrl-C and re-run with different t.
```
rosrun ivr_assignment fk.py t
```

## 3.2
Run target tracker to publish target position.
```
rosrun ivr_assignment target_tracker.py
```

Run end effector tracker to publish end effector position.
```
rosrun ivr_assignment end_effector_tracker.py
```

Run target follower to control the robot in following the target. Provide parameter 'closed' to use closed-loop control as in lab 3.
```
rosrun ivr_assignment target_follower.py close
```

## 4.2
Run target tracker to publish target position.
```
rosrun ivr_assignment target_tracker.py
```

Run end effector tracker to publish end effector position.
```
rosrun ivr_assignment end_effector_tracker.py
```

Run target follower to control the robot in following the target. Provide parameter 'null' to use a null space to avoid the obstacle.
```
rosrun ivr_assignment target_follower.py null
```

## Short description of each module
Executables:
- angles_etimator.py: Execute to estimate the angles of the robot's joints using vision.
- end_effector_tracker.py: Execute to estimate the position of the end effector using vision.
- fk.py: contains functions used for forward kinematics and velocity kinematics of the robot. Execute to run tests on FK matrix.
- image1/2.py: Unused
- joints_mover.py: Execute with parameter \"2.1\" to move the joints according to the equations specified in task 2.1. Parameter \"reset\" resets the robot's joints to zero.
- shape_classifier_trainer.py: Execute to train a new model according to desired parameters using image data. Default directory for image data is "./images/ml_samples/".
- target_follower.py: Execute to run control of the robot for moving the end effector to the target. Specify parameter \"closed\" for closed loop controller and \"null\" for controller using null space.
- target_image_sampler.py: Used once to gather sample images of the data, which were labelled and saved to be used for training the classifier.
- target_move.py: Pre-existing file which is responsible for the movement of the target and the obstacle
- target_tracker.py: Execute to estimate the position of the target and the obstacle

Auxiliary:
- constants.py: Gives access to constants widely used throughout the project.
- coordinates_extractor.py: Contains tools to get position of blobs in an image.
- joints_locator.py: Contains tools for getting joints positions given an image from camera 1 and one from camera 2.
- math_utils.py: Contains math-related utility functions 
- shape_classifier.py: Contains methods for loading, saving, training and predicting shape class using a MGC model.
- target_locator.py: Contains tools for estimating the position of the target and the obstacle when given two images.
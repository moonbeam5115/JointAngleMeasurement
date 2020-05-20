# Using Computer Vision and Transfer Learning to Classify Poses and Measure Joint Angles
While training to increase my athletic performance, I noticed the importance of mobility in decreasing my chance of injury  while performing athletic movements. The ability of our muscles to lengthen and contract rapidly is necessary in order to execute athletic movements such as sprinting and jumping. If our muscles are not accustomed to stretching beyond a certain extent, muscle pulls, strains, or even tears may occur, leading to injuries that force athletes to cease activity. 

My idea with this project is to provide a method for joint angle measurement, which could be used to assess an athlete's mobility. This information could be incorporated into their training program, so as to not include movements or exercises that may cause injury to the athlete. It could also be used to determine which athlete could benefit more from mobility and flexibility routines. Finally, in a rehabilitation setting, the measure of joint angles could be used to measure progress for rehabilitation patients as they perform neuromotor exercises to regain range of motion.

Given an image, the Pose Analyzer does 3 things:
1. Classifies the input image's pose: squatting, bending, or raising arms
2. Overlays the predicted human joint position for the input image
3. Draw and display joint angle information currently only works for specific instances of squatting poses. More work will need to be done to generalize the joint angle measurement for all poses and cases

# Data Sources

* Took 291 personal pictures of the following poses: squatting, bending, and raising arms
* Took 99 pictures from google search for the following poses: squatting, bending, and raising arms

# Background Information

* In 2018, 34.2 million Americans, or 10.5% of the population, had diabetes
* The estimated total economic cost of diagnosed diabetes in 2012 is $245 billion, a 41% increase from our previous estimate of $174 billion (in 2007 dollars)
* Exercise helped to lower insulin resistance in previously sedentary older adults with abdominal obesity at risk for diabetes (Resistance training and aerobic) - Harvard Study
* Cross sectional, prospective and retrospective studies have found significant association between physical inactivity and Type 2 Diabetes

# Requirements to Run This Project

This project requires quite a few libraries and dependencies. It is recommended that you install a docker container as well as a virtual environment in order to run this project.

Option 1: Install base docker container and then install libraries and dependencies
*Instructions to follow

Option 2: (Recommended) Pull docker image with all necessary libraries and dependencies
*Instructions to follow

![Results](/img/pred_result_arm_raise_001.jpg)

*description


![Results](/img/pred_result_arm_raise_002.jpg)
*description   

![Results](/img/pred_result_arm_raise_003.jpg)

*description 

![Results](/img/pred_result_bending_001.jpg)

*description

![Results](/img/pred_result_bending_002.jpg

*description
&nbsp;  
&nbsp;  
&nbsp;  
&nbsp;  
*description

![Results](/img/pred_result_jointAngle_001.jpg)

&nbsp;
&nbsp;

*description

![Results](/img/pred_result_squat_001.jpg)

&nbsp;
&nbsp;
 
*description

![Results](/img/pred_result_squat_002.jpg)

&nbsp;
&nbsp;

*description

![Results](/img/pred_result_squat_003.jpg)

&nbsp;
&nbsp;

*description

# Conlusion

* More studies should be conducted to analyze the impact of lifestyle on type 2 Diabetes: Exercise, Sleep, Nutrition, Stress Levels, Drug Use
* Exercise can lower insulin resistance and should be incorporated into a holistic treatment for people suffering from diabetes
* Although the relationship between nutrition and diabetes is complex, the most harmful products to consume are carbonated beverages
* Other factors like lack of sleep seem to have a high correlation with diabetes incidence - why?

# Next Steps
* Gather more accurate information about lifestyle choices

* Gather information about economic burden due to diabetes

* Figure out what kind of effect exercise and better nutrition would have on diabetes related healthcare costs

* Adjust the analysis to include the population size

# References
*description

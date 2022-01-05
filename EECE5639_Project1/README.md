# EECE 5639 CV Project 1

# Notes on running this code

You must put the Office data set and/or RedChair data set provided with the assignment in the same directory as these scripts.

Run thresh_est.py to generate a suggested threshold range from the noise of a data set. It is currently set to look at the Office data set.

Run spatial_smoothing.py to generate various smoothed versions of a data set. It is currently set up to generate 6 variations of smoothed data for the Office data set.

Run temporal_difference.py and/or derivative_of_gaussian.py if you wish to run a single instance of our TD or DOG filter.

Run main.py to accomplish experiments outlined in the assignment. It has several parameters which can be set to tailor the experiment (choose a task, choose a data set, show playback video, save a test frame)

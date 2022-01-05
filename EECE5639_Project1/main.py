import os
from frameload import *
from temporal_difference import *
from derivative_of_gaussian import *
from spatial_smoothing import *

office = os.path.dirname(__file__) + '/Office/img01_'
redchair = os.path.dirname(__file__) + "/RedChair/advbgst1_21_"

office_box3x3 = os.path.dirname(__file__) + '/Office/box3x3/img01_'
office_box5x5 = os.path.dirname(__file__) + '/Office/box5x5/img01_'
office_gauss10 = os.path.dirname(__file__) + '/Office/gaussian1.0/img01_'
office_gauss14 = os.path.dirname(__file__) + '/Office/gaussian1.4/img01_'
office_gauss18 = os.path.dirname(__file__) + '/Office/gaussian1.8/img01_'
office_gauss22 = os.path.dirname(__file__) + '/Office/gaussian2.2/img01_'

redchair_box3x3 = os.path.dirname(__file__) + "/RedChair/box3x3/advbgst1_21_"
redchair_box5x5 = os.path.dirname(__file__) + "/RedChair/box5x5/advbgst1_21_"
redchair_gauss10 = os.path.dirname(__file__) + "/RedChair/gaussian1.0/advbgst1_21_"
redchair_gauss14 = os.path.dirname(__file__) + "/RedChair/gaussian1.4/advbgst1_21_"
redchair_gauss18 = os.path.dirname(__file__) + "/RedChair/gaussian1.8/advbgst1_21_"
redchair_gauss22 = os.path.dirname(__file__) + "/RedChair/gaussian2.2/advbgst1_21_"

office_set = [office, office_box3x3, office_box5x5, office_gauss10, office_gauss14, office_gauss18, office_gauss22]
redchair_set = [redchair, redchair_box3x3, redchair_box5x5, redchair_gauss10, redchair_gauss14, redchair_gauss18, redchair_gauss22]


def main():
    # TODO make these into command line parameters
    task=3
    show_video=False
    save_frame=57
    video_set = office_set
    video = video_set[0]

    # A study of the effect of the range of temporal gradiant on motion detection
    if task==1:
        temp_diff_filter(video=video, show_video=show_video, save_frame=save_frame)
        for sig in [1.0, 1.4, 1.8, 2.2]:
            dog_filter(video=video, tsigma=sig, show_video=show_video, save_frame=save_frame)

    # A study of the effect of spatial smoothing on motion detection
    if task==2:
        for video in video_set:
            temp_diff_filter(video=video, show_video=show_video, save_frame=save_frame)

    # A study of the effect of thresholding on motion detection
    if task==3:
        for thresh in range(21):
            temp_diff_filter(video=video, thresh=thresh, show_video=show_video, save_frame=save_frame)

    
if __name__ == "__main__":
    main()

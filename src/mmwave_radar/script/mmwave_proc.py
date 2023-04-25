#!/usr/bin/env python3
import rospy
from mmwave_radar.msg import adcData
import pickle


cnt = 1
# pub = rospy.Publisher("mmwave_radar_point_cloud")
def gen_pub_point_cloud(adcData):
    global cnt
    with open(f'adcData_{cnt}.pkl', 'wb') as f:
        pickle.dump(adcData.data, f)
    cnt += 1


if __name__ == "__main__":
    rospy.init_node("mmwave_radar_subscriber")
    sub = rospy.Subscriber("mmwave_radar_raw_data", adcData, gen_pub_point_cloud, queue_size=10)
    rospy.spin()

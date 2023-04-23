import rospy
from mmwave_radar.msg import adcData


# pub = rospy.Publisher("mmwave_radar_point_cloud")
def gen_pub_point_cloud(adcData):
    print(type(adcData))
    print(type(adcData.header))
    print(type(adcData.size))
    print(type(adcData.data))


if __name__ == "__main__":
    rospy.init_node("mmwave_radar_subscriber")
    sub = rospy.Subscriber("mmwave_radar_raw_data", adcData, gen_pub_point_cloud, queue_size=10)
    rospy.spin()

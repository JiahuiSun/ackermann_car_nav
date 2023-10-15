import globalvar as gl
import rospy
from webots_ros.msg import BoolStamped

def rostopic_touch_sensor_get_callback(msg):
    gl.set_value('touch', msg.data)

def rostopic_touch_sensor_get():
    rospy.init_node('touch_sensor_listener', anonymous=True)
    rospy.Subscriber("/vehicle/touch_sensor/value", BoolStamped, rostopic_touch_sensor_get_callback)
    rospy.spin()

if __name__ == "__main__":
    rostopic_touch_sensor_get()
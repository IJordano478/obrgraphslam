import numpy as np
import matplotlib.pyplot as plt

blue_cones = np.array([(4.0, 1.0),
                       (4.0, 2.0),
                       (4.0, 3.0),
                       (4.0, 4.0),
                       (4.0, 5.0),
                       (4.0, 6.0),
                       (3.0, 7.0),
                       (2.0, 8.0),
                       (1.0, 8.0),
                       (0.0, 8.0),
                       (-1.0, 8.0),
                       (-2.0, 8.0),
                       (-3.0, 7.0),
                       (-4.0, 6.0),
                       (-4.0, 5.0),
                       (-4.0, 4.0),
                       (-4.0, 3.0),
                       (-4.0, 2.0),
                       (-4.0, 1.0),
                       (-4.0, 0.0),
                       (-4.0, -1.0),
                       (-4.0, -2.0),
                       (-4.0, -3.0),
                       (-4.0, -4.0),
                       (-4.0, -5.0),
                       (-4.0, -6.0),
                       (-3.0, -7.0),
                       (-2.0, -8.0),
                       (-1.0, -8.0),
                       (0.0, -8.0),
                       (1.0, -8.0),
                       (2.0, -8.0),
                       (3.0, -7.0),
                       (4.0, -6.0),
                       (4.0, -5.0),
                       (4.0, -4.0),
                       (4.0, -3.0),
                       (4.0, -2.0),
                       (4.0, -1.0)])

yellow_cones = np.array([(2.0, 1.0),
                         (2.0, 2.0),
                         (2.0, 3.0),
                         (2.0, 4.0),
                         (2.0, 5.0),
                         (1.0, 6.0),
                         (0.0, 6.0),
                         (-1.0, 6.0),
                         (-2.0, 5.0),
                         (-2.0, 4.0),
                         (-2.0, 3.0),
                         (-2.0, 2.0),
                         (-2.0, 1.0),
                         (-2.0, 0.0),
                         (-2.0, -1.0),
                         (-2.0, -2.0),
                         (-2.0, -3.0),
                         (-2.0, -4.0),
                         (-2.0, -5.0),
                         (-1.0, -6.0),
                         (0.0, -6.0),
                         (1.0, -6.0),
                         (2.0, -5.0),
                         (2.0, -4.0),
                         (2.0, -3.0),
                         (2.0, -2.0),
                         (2.0, -1.0)])

orange_cones = np.array([(2.0, 0.0),
                         (4.0, 0.0)])


plt.plot(blue_cones[:, 0], blue_cones[:, 1], '^b', label='Blue cones')
plt.plot(yellow_cones[:, 0], yellow_cones[:, 1], '^y', label='Yellow cones')
plt.plot(orange_cones[:, 0], orange_cones[:, 1],
         marker='x', c='orange', linestyle='None', label='Starting line')

plt.legend(bbox_to_anchor=(1.05, 1), fontsize='small')
plt.title('Oval Track')
plt.xlabel('X distance (m)')
plt.ylabel('Y distance (m)')
plt.gca().set_aspect('equal')
plt.show()
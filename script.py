import numpy as np
import matplotlib.pyplot as plt
import math

class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __abs__(self):
        return (self.x**2 + self.y**2 + self.z**2)**(1/2)

    def normalize(self):
        magnitude = abs(self)
        if magnitude != 0:
            self.x /= magnitude
            self.y /= magnitude
            self.z /= magnitude

    def __mul__(self, other):
        if isinstance(other,Vec3):
            return self.x*other.x + self.y*other.y + self.z*other.z
        else:
            return Vec3(self.x * other, self.y * other, self.z * other)


class Sphere:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
    def intersect(self, ray_origin : Vec3, ray_direction : Vec3):
        ray_direction.normalize()
        origin_to_center = self.center - ray_origin
        proj_length = ray_direction * origin_to_center
        if proj_length > 0:
            proj_point = ray_origin + ray_direction * proj_length
            if abs(proj_point - self.center) <= self.radius:
                return True
        return False

def write_ppm(image_matrix: np.ndarray, image_file):
    width = image_matrix.shape[1]
    height = image_matrix.shape[0]

    with open(image_file, 'w') as f:
        f.write("P3\n")
        f.write(f"{width} {height}\n")
        f.write(f"255\n")
        for i in range(height):
            for j in range(width):
                f.write(f"{image_matrix[i][j][0]} {image_matrix[i][j][1]} {image_matrix[i][j][2]}\n")
        f.close()

scene_objects = [Sphere(Vec3(0.0, 0.0, -8.0), 4)]

def render(scene_objects, img_w, img_h):
    img_matrix = np.ndarray((img_h,img_w, 3), dtype=np.uint8)
    fov = (170.0/360) * (2*math.pi)
    screen_dist = 1.0
    up_dir = (math.tan(fov/2) * screen_dist) / (img_h / 2.0)
    right_dir = (math.tan(fov/2) * screen_dist) / (img_w / 2.0)
    for i in range(img_h):
        for j in range(img_w):
            cam_direction = Vec3(right_dir * ((j + 0.5) - img_w/2),
                                 up_dir * (img_h/2 - (i + 0.5)), -1.0) 
            cam_direction.normalize()
            for obj in scene_objects:
                if (obj.intersect(Vec3(0.0,0.0,0.0), cam_direction)):
                    img_matrix[i][j] = [255,0,0]
                else:
                    img_matrix[i][j] = [0.0, 0.0, 0.0]
    return img_matrix

img_matrix = render(scene_objects, 1024, 768)
write_ppm(img_matrix, "img.ppm")